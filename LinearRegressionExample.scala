import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{DenseVector, Matrices}
import breeze.linalg.{DenseMatrix, inv}

val conf = new SparkConf().setAppName("LinearRegressionExample").setMaster("local[*]")
val sc = new SparkContext(conf)
val rowsX = sc.parallelize(Seq(
  DenseVector(1.0, 2.0),
  DenseVector(3.0, 4.0)
))
val matX = Matrices.dense(2, 2, rowsX.flatMap(_.toArray).collect())

val vecY = sc.parallelize(Seq(5.0, 6.0))
val y = vecY.collect()
val XTX = matX.transpose.multiply(matX)
val XTXBreeze = new DenseMatrix[Double](XTX.numRows, XTX.numCols, XTX.toArray)
val XTXInverse = inv(XTXBreeze)
val XTY = matX.transpose.multiply(new DenseVector(y))
val theta = new DenseVector[Double](XTXInverse * XTY.toArray)
// Test data for gradient descent
val trainingData = sc.parallelize(Seq(
  (DenseVector(1.0, 2.0), 5.0),
  (DenseVector(3.0, 4.0), 6.0)
))

val initialTheta = DenseVector.zeros[Double](2)
val alpha = 0.001
val numIterations = 5

val (finalWeights, trainingErrors) = gradientDescent(numIterations, alpha, initialTheta, trainingData)
println(s"Final weights: $finalWeights")
println(s"Training errors: ${trainingErrors.mkString(", ")}")
