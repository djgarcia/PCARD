package org.apache.spark.ml.classification

import org.apache.spark.annotation.{Experimental, Since}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector => ML}
import org.apache.spark.ml.param._
import org.apache.spark.ml.tree.{PCARDEnsembleModel, TreeClassifierParams}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable, MetadataUtils}
import org.apache.spark.ml.{PipelineModel, PredictorParams}
import org.apache.spark.mllib.feature.PCAModel
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.{PCARD => OldPCARD, PCARDModel => OldPCARDModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset

private[ml] trait PCARDParams extends PredictorParams {

  final val nTrees: IntParam = new IntParam(this, "nTrees", "Number of trees.",
    ParamValidators.gtEq(1))

  final def getTrees: Int = $(nTrees)

  final val cuts: IntParam = new IntParam(this, "cuts", "Number of intervals.",
    ParamValidators.gtEq(1))

  final def getCuts: Int = $(cuts)
}

@Since("1.5.0")
@Experimental
class PCARDClassifier @Since("1.5.0")(@Since("1.5.0") override val uid: String)
  extends ProbabilisticClassifier[ML, PCARDClassifier, PCARDClassificationModel]
    with PCARDParams with TreeClassifierParams {

  @Since("1.5.0")
  def this() = this(Identifiable.randomUID("pcard"))

  @Since("1.5.0")
  def setTrees(value: Int): this.type = set(nTrees, value)

  setDefault(nTrees -> 10)

  @Since("1.5.0")
  def setCuts(value: Int): this.type = set(cuts, value)

  setDefault(cuts -> 5)

  override protected def train(dataset: Dataset[_]): PCARDClassificationModel = {
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))
    val numClasses: Int = MetadataUtils.getNumClasses(dataset.schema($(labelCol))) match {
      case Some(n: Int) => n
      case None => throw new IllegalArgumentException("PCARDClassifier was given input" +
        s" with invalid label column ${$(labelCol)}, without the number of classes" +
        " specified. See StringIndexer.")
    }

    val oldDataset: RDD[LabeledPoint] = extractLabeledPoints(dataset).map { x => LabeledPoint(x.label, Vectors.dense(x.features.toArray)) }
    val oldModel = OldPCARD.train(oldDataset, $(nTrees), $(cuts))
    PCARDClassificationModel.fromOld(oldModel, this)
    //oldModel.asInstanceOf[PCARDClassificationModel]
  }

  @Since("1.5.0")
  override def copy(extra: ParamMap): PCARDClassifier = defaultCopy(extra)
}

@Since("1.6.0")
object PCARDClassifier extends DefaultParamsReadable[PCARDClassifier] {

  @Since("1.6.0")
  override def load(path: String): PCARDClassifier = super.load(path)
}

@Since("1.5.0")
@Experimental
class PCARDClassificationModel private[ml](override val uid: String,
                                           override val nTrees: Int,
                                           override val disc: Array[Array[Array[Double]]],
                                           override val pcaList: Array[PCAModel],
                                           override val models: Array[PipelineModel],
                                           override val labelsInd: Array[String])
  extends ProbabilisticClassificationModel[ML, PCARDClassificationModel]
    with PCARDEnsembleModel with Serializable {

  @Since("1.5.0")
  override val numFeatures: Int = disc(0).length

  @Since("1.5.0")
  override val numClasses: Int = labelsInd.length

  private[ml] def this(nTrees: Int,
                       disc: Array[Array[Array[Double]]],
                       pcaList: Array[PCAModel],
                       models: Array[PipelineModel],
                       labelsInd: Array[String]) =
    this(Identifiable.randomUID("pcardc"), nTrees, disc, pcaList, models, labelsInd)

  override protected def predict(features: ML): Double = {
    val model = new OldPCARDModel(nTrees, disc, pcaList, models, labelsInd)
    model.predict(Vectors.dense(features.toArray))
  }

  override protected def predictRaw(features: ML): ML = {
    val model = new OldPCARDModel(nTrees, disc, pcaList, models, labelsInd)
    val pred = Array.fill(numClasses)(0.0)
    pred(model.predict(Vectors.dense(features.toArray)).toInt) = numFeatures
    Vectors.dense(pred).asML
  }

  override protected def raw2probabilityInPlace(rawPrediction: ML): ML = {
    rawPrediction match {
      case dv: DenseVector =>
        ProbabilisticClassificationModel.normalizeToProbabilitiesInPlace(dv)
        dv
      case sv: SparseVector =>
        throw new RuntimeException("Unexpected error in PCARDClassificationModel:" +
          " raw2probabilityInPlace encountered SparseVector")
    }
  }

  @Since("1.5.0")
  override def copy(extra: ParamMap): PCARDClassificationModel = {
    copyValues(new PCARDClassificationModel(uid, nTrees: Int,
      disc: Array[Array[Array[Double]]],
      pcaList: Array[PCAModel],
      models: Array[PipelineModel],
      labelsInd: Array[String]), extra)
  }

  @Since("1.5.0")
  override def toString: String = {
    s"PCARDClassificationModel (uid=$uid) with $nTrees trees"
  }

  private[ml] def toOld: OldPCARDModel = {
    new OldPCARDModel(nTrees, disc, pcaList, models, labelsInd)
  }
}

private[ml] object PCARDClassificationModel {
  def fromOld(oldModel: OldPCARDModel, parent: PCARDClassifier): PCARDClassificationModel = {
    val uid = if (parent != null) parent.uid else Identifiable.randomUID("nb")
    val nTrees = oldModel.getTrees
    val cuts = oldModel.getCuts
    val pcaList = oldModel.getPcaList
    val models = oldModel.getModels
    val labelsInd = oldModel.getLabelsInd
    new PCARDClassificationModel(uid, nTrees, cuts, pcaList, models, labelsInd)
  }
}
