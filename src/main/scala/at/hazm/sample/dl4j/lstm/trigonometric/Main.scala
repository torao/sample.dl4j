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

  case class Sample(theta:Double) {
    // lazy val sin:Double = math.sin(theta)
    lazy val cos:Double = math.cos(theta)
    // lazy val tan:Double = math.tan(theta)
    lazy val features:Seq[Double] = Seq(cos)
  }

  object Sample {
    def features:Int = 1
  }

  /** (0, 2π] の分割数。 */
  val resolution:Int = 1000

  /** 事前計算済みの学習/テスト用サンプルデータ。 */
  val samples:Seq[Sample] = for(i <- 0 until resolution) yield Sample(2 * math.Pi * i / resolution)

  def main(args:Array[String]):Unit = {

    //Length of each training example sequence to use. This could certainly be increased
    val tbpttLength = 50
    //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
    val numEpochs = 500 // 学習時のエポック数
    val lstmLayerSize = Sample.features * 2 // 隠れ層のノード数
    val miniBatchSize = 35
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
        .nIn(Sample.features)
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
        .nOut(Sample.features)
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
    val dataSets = Iterator.from(0).takeWhile(_ * miniBatchSize * sampleLength < sampleLength).map { i =>
      createTrainingDataSet(i, miniBatchSize, sampleLength)
    }.toList
    for(_ <- 1 to numEpochs) {
      dataSets.foreach { ds => net.fit(ds) }
    }

    val testInputArray = Nd4j.zeros(Sample.features)
    for(j <- 0 until Sample.features) testInputArray.putScalar(j, math.cos(0))
    net.rnnClearPreviousState()
    for(_ <- 0 until 200){
      val outputArray = net.rnnTimeStep(testInputArray)
      println((for(i <- 0 until Sample.features) yield outputArray.getDouble(i)).mkString("\t"))
    }
    println("----------------")

    // (0, 2π] の範囲で任意の θ を開始点とし π を入力して続きの π を推測する
    val dθ = 2 * math.Pi / resolution
    val θ0 = rand.nextDouble() * 2 * math.Pi
    val predictIn = createPredictionDataSequence(0, miniBatchSize, sampleLength)
    net.rnnClearPreviousState()
    val predictOut = net.rnnTimeStep(predictIn)

    val output = predictOut.tensorAlongDimension(predictOut.size(2)-1, 1, 0)

    System.out.println(s"θ\tsin(θ)\tcos(θ)\ttan(θ)")
    for(k <- 0 until predictOut.shape().apply(2)) {
      val θ = θ0 + dθ * k
      System.out.println(s"${(0 until Sample.features).map{ j => predictOut.getDouble(0, j, k)}.mkString("\t")}")
    }
  }

  val rand = new Random(99999)

  // 学習用データの作成
  // (sin(θ), cos(θ), tan(θ)):[3次元] を入力、(sin(θ+dθ), cos(θ+dθ), tan(θ+dθ)):[3次元] を出力とする。ここで Θ∈(0, 2π] とする。
  private[this] def createTrainingDataSet(i:Int, miniBatchSize:Int, exampleLength:Int):DataSet = {
    if(i * miniBatchSize * exampleLength >= samples.length - 1) {
      throw new IllegalArgumentException(s"$i x $miniBatchSize x $exampleLength >= ${samples.length}")
    }

    /*
    val possibleFullExampleLength = math.min((samples.length - 1 - i * miniBatchSize * exampleLength) / miniBatchSize, exampleLength)
    val t0 = i * miniBatchSize * exampleLength
    val (actualExampleLength, actualBatchSize) = if(possibleFullExampleLength == 0) {
      (1, (samples.length - 1) % miniBatchSize)
    } else {
      (possibleFullExampleLength, miniBatchSize)
    }

    val in = Nd4j.zeros(Array(actualBatchSize, Sample.features, actualExampleLength), 'f')
    val out = Nd4j.zeros(Array(actualBatchSize, Sample.features, actualExampleLength), 'f')
    for(i <- 0 until actualBatchSize; k <- 0 until actualExampleLength) {
      for(j <- 0 until Sample.features){
        in.putScalar(Array(i, j, k), samples(t0 + k * miniBatchSize + i).features(j))
        out.putScalar(Array(i, j, k), samples(t0 + k * miniBatchSize + i + 1).features(j))
      }
    }
    */
    val t0 = i * miniBatchSize * exampleLength
    val length = math.min(samples.length, exampleLength)
    val in = Nd4j.zeros(Array(1, Sample.features, length), 'f')
    val out = Nd4j.zeros(Array(1, Sample.features, length), 'f')
    for(k <- 0 until length) {
      for(j <- 0 until Sample.features){
        in.putScalar(Array(0, j, k), samples(t0 + k).features(j))
        out.putScalar(Array(0, j, k), samples(t0 + k + 1).features(j))
      }
    }
    new DataSet(in, out)
  }

  private[this] def createPredictionDataSequence(offset:Int, miniBatchSize:Int, exampleLength:Int):INDArray = {
    val predictIn = Nd4j.zeros(Array(miniBatchSize, Sample.features, exampleLength), 'f')
    for(i <- 0 until miniBatchSize; k <- 0 until exampleLength) {
      for(j <- 0 until Sample.features){
        predictIn.putScalar(Array(i, j, k), samples((k * miniBatchSize + i) % samples.length).features(j))
      }
    }
    predictIn
  }

}