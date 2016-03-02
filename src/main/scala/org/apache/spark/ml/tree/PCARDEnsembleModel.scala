package org.apache.spark.ml.tree

import org.apache.spark.ml.PipelineModel
import org.apache.spark.mllib.feature.PCAModel

/**
  * Abstraction for PCARD Ensemble models.
  *
  * TODO: Add support for predicting probabilities and raw predictions  SPARK-3727
  */
private[ml] trait PCARDEnsembleModel {

  def nTrees: Int
  def disc: Array[Array[Array[Double]]]
  def pcaList: Array[PCAModel]
  def models: Array[PipelineModel]
  def labelsInd: Array[String]

  /** Summary of the model */
  override def toString: String = {
    // Implementing classes should generally override this method to be more descriptive.
    s"PCARDModel with $nTrees trees"
  }

  /** Full description of model */
  def toDebugString: String = {
    val header = toString + "\n"
    header
  }
}