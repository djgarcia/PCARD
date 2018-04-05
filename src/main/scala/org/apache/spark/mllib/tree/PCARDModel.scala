package org.apache.spark.mllib.tree

import org.apache.spark.SparkContext
import org.apache.spark.ml.PipelineModel
import org.apache.spark.mllib.feature.PCAModel
import org.apache.spark.mllib.linalg.{DenseMatrix, Vector, Vectors}
import org.apache.spark.ml.linalg.{Vector => ML}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}

class PCARDModel(val nTrees: Int, val disc: Array[Array[Array[Double]]], val pcaList: Array[PCAModel], val models: Array[PipelineModel], val labelsInd: Array[String]) extends Serializable {

  private def assignDiscreteValue(value: Double, thresholds: Seq[Double]): Double = {
    if (thresholds.isEmpty) {
      value
    } else {
      val ret = thresholds.indexWhere {
        value <= _
      }
      if (ret == -1) {
        thresholds.size.toDouble
      } else {
        ret.toDouble
      }
    }
  }

  def getTrees: Int = nTrees

  def getCuts: Array[Array[Array[Double]]] = disc

  def getPcaList: Array[PCAModel] = pcaList

  def getModels: Array[PipelineModel] = models

  def getLabelsInd: Array[String] = labelsInd

  def predict(test: RDD[LabeledPoint]): Array[Double] = {

    val sqlContext = new org.apache.spark.sql.SQLContext(test.context)
    import sqlContext.implicits._

    val tama = test.count.toInt
    val totalPredictions = Array.fill(tama)(Array.fill(labelsInd.length)(0.0))
    val dataTest = test.map(_.features)

    for (c <- 0 until nTrees) {

      //RD
      val discTest = dataTest.map { l =>
        val features = l.toArray
        val newValues = for (i <- features.indices)
          yield assignDiscreteValue(features(i), disc(c)(i).toSeq)
        Vectors.dense(newValues.toArray)
      }
      val featDisc = discTest.map(l => l.toArray)

      //PCA

      val pcaData = test.map(p => LabeledPoint(p.label, pcaList(c).transform(p.features)))

      //PCARD

      val PCARD = pcaData.zip(featDisc).map(l => LabeledPoint(l._1.label, Vectors.dense(l._2 ++ l._1.features.toArray)).asML).toDF()

      val predictions = models(c).transform(PCARD).select("probability").collect()
      var i = 0
      predictions.foreach {
        case Row(prob: ML) =>
          totalPredictions(i) = Array(totalPredictions(i), prob.toArray).transpose.map(_.sum)
          i += 1
      }
    }
    val pos = totalPredictions.zipWithIndex.map { case (k, v) => k.indexOf(k.max).toDouble }
    pos.map(l => labelsInd(l.toInt).toDouble)
  }

  def predict(data: Vector): Double = {

    val sqlContext = SparkSession.builder().getOrCreate()
    import sqlContext.implicits._

    var totalPredictions = Array.fill(labelsInd.length)(0.0)

    for (c <- 0 until nTrees) {

      //RD
      val features = data.toArray
      val newValues = for (i <- features.indices)
        yield assignDiscreteValue(features(i), disc(c)(i).toSeq)
      val featDisc = Vectors.dense(newValues.toArray)

      //PCA

      val pcaData = pcaList(c).transform(data)

      //PCARD

      val PCARD = Vectors.dense(featDisc.toArray ++ pcaData.toArray)

      val test = SparkContext.getOrCreate().parallelize(Seq(LabeledPoint(0, PCARD).asML)).toDF()

      val predictions = models(c).transform(test).select("probability").collect()
      var i = 0
      predictions.foreach {
        case Row(prob: ML) =>
          totalPredictions = (totalPredictions, prob.toArray).zipped.map(_ + _)
          i += 1
      }
    }
    val pos = totalPredictions.indexOf(totalPredictions.max)
    labelsInd(pos).toDouble
  }

  def getWeights(): Array[DenseMatrix] = {
    val weights: Array[DenseMatrix] = new Array[DenseMatrix](nTrees)

    for (c <- 0 until nTrees) {
      weights(c) = pcaList(c).pc
    }
    weights
  }
}
