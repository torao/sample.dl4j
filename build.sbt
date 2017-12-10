organization := "at.hazm"

name := "sample.dl4j"

version := "1.0.0"

scalaVersion := "2.12.4"

libraryDependencies ++= Seq(
  "org.deeplearning4j" % "deeplearning4j-nlp" % "0.9.+",
  "org.nd4j" % "nd4j-native-platform" % "0.9.+",      // GPU を使用する場合はここを変更
  "org.slf4j" % "slf4j-log4j12" % "1.7.25"
)
