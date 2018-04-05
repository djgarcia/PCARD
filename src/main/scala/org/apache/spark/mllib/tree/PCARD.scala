package org.apache.spark.mllib.tree

import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.feature.{PCA, PCAModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd._

import scala.util.Random

class PCARD private(val data: RDD[LabeledPoint], val nTrees: Int, val cuts: Int) extends Serializable {

  private val disc: Array[Array[Array[Double]]] = new Array[Array[Array[Double]]](nTrees)
  private val pcaList: Array[PCAModel] = new Array[PCAModel](nTrees)
  private val models: Array[PipelineModel] = new Array[PipelineModel](nTrees)
  private var labelsInd: Array[String] = new Array[String](1)

  private def assignDiscreteValue(value: Double, thresholds: Seq[Double]) = {
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

  def runTrain(): PCARDModel = {

    val sqlContext = new org.apache.spark.sql.SQLContext(data.context)
    import sqlContext.implicits._

    for (c <- 0 until nTrees) {

      //RD
      val cortes = Seq.fill(cuts - 1)(Random.nextInt(data.count.toInt))
      val indexKey = data.zipWithIndex.map { case (k, v) => (v, k) }
      val temp = new Array[Array[Double]](cortes.size)

      for (i <- cortes.indices) {
        temp(i) = indexKey.lookup(cortes(i)).head.features.toArray
      }

      val trans = temp.transpose.zipWithIndex.map {
        case (l, i) =>
          val feat = l.sorted.distinct
          if (feat.length == 1) {
            val col = data.map(l => l.features.toArray.slice(i, i + 1)).collect.flatten.distinct
            (Random.shuffle(col.toList) take cuts - 1).sorted.toArray
          } else {
            feat
          }
      }

      disc(c) = trans

      val discData = data.map { l =>
        val features = l.features.toArray
        val newValues = for (i <- features.indices)
          yield assignDiscreteValue(features(i), trans(i).toSeq)
        LabeledPoint(l.label, Vectors.dense(newValues.toArray))
      }

      //PCA

      val tam = data.first().features.size

      val rnd = new scala.util.Random
      val range = 1 to tam - 1
      val size = range(rnd.nextInt(range.length))

      val pca = new PCA(size).fit(data.map(_.features))
      pcaList(c) = pca
      val pcaData = data.map(p => p.copy(features = pca.transform(p.features)))

      //PCARD
      val PCARD = pcaData.zip(discData).map(l => LabeledPoint(l._1.label, Vectors.dense(l._2.features.toArray ++ l._1.features.toArray)).asML).toDF()

      val labelIndexer = new StringIndexer()
        .setInputCol("label")
        .setOutputCol("indexedLabel")
        .fit(PCARD)

      labelsInd = labelIndexer.labels

      val dt = new DecisionTreeClassifier()
        .setLabelCol("indexedLabel")
        .setFeaturesCol("features")
      //.setMaxMemoryInMB(5000)

      val pipeline = new Pipeline()
        .setStages(Array(labelIndexer, dt))

      models(c) = pipeline.fit(PCARD)
    }
    new PCARDModel(nTrees, disc, pcaList, models, labelsInd)
  }
}

object PCARD {
  def train(input: RDD[LabeledPoint], nTrees: Int, cuts: Int): PCARDModel = {
    new PCARD(input, nTrees, cuts).runTrain()
  }
}
