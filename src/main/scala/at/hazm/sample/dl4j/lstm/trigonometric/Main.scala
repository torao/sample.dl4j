package at.hazm.sample.dl4j.lstm.trigonometric

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{BackpropType, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.LoggerFactory

import scala.util.Random

/**
  * 三角関数 y = (sin x, cos x, tan x) のベクトルを学習させて続きを予測させるサンプルコード。
  */
object Main {
  private[this] val logger = LoggerFactory.getLogger(getClass.getName.dropRight(1))

  case class Data(theta:Double, sin:Double, cos:Double, tan:Double)

  def main(args:Array[String]):Unit = {

    //Length of each training example sequence to use. This could certainly be increased
    val tbpttLength = 50
    //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
    val numEpochs = 2000 // 学習時のエポック数
    val features = 3 // ベクトル (sin, cos, tan) の長さ
    val lstmLayerSize = features * 2 // 隠れ層のノード数
    val miniBatchSize = 100
    val sampleLength = 50

    // LSTM ネットワークの作成
    val conf = new NeuralNetConfiguration.Builder()
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .learningRate(0.1)
      .seed(12345)
      .regularization(true)
      .l2(0.001)
      .weightInit(WeightInit.XAVIER)
      .updater(Updater.RMSPROP)
      .list
      .layer(0, new GravesLSTM.Builder()
        .nIn(features)
        .nOut(lstmLayerSize)
        .activation(Activation.TANH)
        .build
      )
      .layer(1, new GravesLSTM.Builder()
        .nIn(lstmLayerSize)
        .nOut(lstmLayerSize)
        .activation(Activation.TANH)
        .build()
      )
      .layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT) //MCXENT + softmax for classification
        .activation(Activation.SOFTMAX)
        .nIn(lstmLayerSize)
        .nOut(features)
        .build()
      )
      .backpropType(BackpropType.TruncatedBPTT)
      .tBPTTForwardLength(tbpttLength)
      .tBPTTBackwardLength(tbpttLength)
      .pretrain(false)
      .backprop(true)
      .build()

    val net = new MultiLayerNetwork(conf)
    net.init()
    net.setListeners(new ScoreIterationListener(1))

    // レイヤーごとのパラメータをログ出力
    net.getLayers.zipWithIndex.foreach { case (layer, i) =>
      logger.info(f"[$i] numParams = ${layer.numParams()}%,d")
    }
    logger.info(f"total number of network parameters: ${net.getLayers.map(_.numParams).sum}%,d")

    // 学習の実行
    val rand = new Random(99999)
    val dθ = 2 * math.Pi / exampleLength
    for(epoch <- 1 to numEpochs) {
      logger.info(s"[EPOCH $epoch]")

      // 学習用データの作成
      // (sin(θ), cos(θ), tan(θ)):[3次元] を入力、(sin(θ+dθ), cos(θ+dθ), tan(θ+dθ)):[3次元] を出力とする。ここで Θ∈(0, 2π] とする。
      val dataSet = createDataSet(miniBatchSize, sampleLength)
      net.fit(dataSet)
    }

    // (0, 2π] の範囲で任意の θ を開始点とし π を入力して続きの π を推測する
    val θ0 = rand.nextDouble() * 2 * math.Pi
    val predictIn = createPredictionDataSequence(0, sampleLength)
    net.rnnClearPreviousState()
    val predictOut = net.rnnTimeStep(predictIn)

    System.out.println(s"θ\tsin(θ)\tcos(θ)\ttan(θ)")
    for(k <- 0 until predictOut.shape().apply(2)) {
      val θ = θ0 + dθ * k
      val sin = predictOut.getDouble(0, 0, k)
      val cos = predictOut.getDouble(0, 1, k)
      val tan = predictOut.getDouble(0, 2, k)
      System.out.println(s"${(θ + dθ) % (2 * math.Pi)}\t$sin\t$cos\t$tan")
    }
  }

  val rand = new Random(99999)
  val exampleLength:Int = 50 //
  val samples:Seq[Data] = for(i <- 0 until exampleLength) yield {
    val theta = 2 * math.Pi * i / exampleLength
    Data(theta, math.sin(theta), math.cos(theta), math.tan(theta))
  }


  // 学習用データの作成
  // (sin(θ), cos(θ), tan(θ)):[3次元] を入力、(sin(θ+dθ), cos(θ+dθ), tan(θ+dθ)):[3次元] を出力とする。ここで Θ∈(0, 2π] とする。
  private[this] def createDataSet(miniBatchSize:Int, sampleLength:Int):DataSet = {
    val in = Nd4j.zeros(Array(miniBatchSize, 3, sampleLength), 'f')
    val out = Nd4j.zeros(Array(miniBatchSize, 3, sampleLength), 'f')
    for(i <- 0 until miniBatchSize) {
      val offset = rand.nextInt(samples.length)
      for(k <- 0 until sampleLength) {
        in.putScalar(Array(i, 0, k), samples((offset + k) % samples.length).sin)
        in.putScalar(Array(i, 1, k), samples((offset + k) % samples.length).cos)
        in.putScalar(Array(i, 2, k), samples((offset + k) % samples.length).tan)
        out.putScalar(Array(i, 0, k), samples((offset + k + 1) % samples.length).sin)
        out.putScalar(Array(i, 1, k), samples((offset + k + 1) % samples.length).cos)
        out.putScalar(Array(i, 2, k), samples((offset + k + 1) % samples.length).tan)
      }
    }
    new DataSet(in, out)
  }

  private[this] def createPredictionDataSequence(offset:Int, sampleLength:Int):INDArray = {
    val predictIn = Nd4j.zeros(Array(1, 3, sampleLength), 'f')
    for(k <- 0 until sampleLength) {
      predictIn.putScalar(Array(0, 0, k), samples((offset + k) % samples.length).sin)
      predictIn.putScalar(Array(0, 1, k), samples((offset + k) % samples.length).cos)
      predictIn.putScalar(Array(0, 2, k), samples((offset + k) % samples.length).tan)
    }
    predictIn
  }

}