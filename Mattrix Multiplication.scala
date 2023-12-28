import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{DenseVector, Matrices}
import breeze.linalg.{DenseMatrix, inv}
import org.apache.spark.rdd.RDD

object LinearRegressionExample {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("LinearRegressionExample").setMaster("local[*]")
    val sc = new SparkContext(conf)

    // Create RDDs for X and y

    val rowsX = sc.parallelize(Seq(
      DenseVector(1.0, 2.0),
      DenseVector(3.0, 4.0)
    ))
    val matX = Matrices.dense(2, 2, rowsX.flatMap(_.toArray).collect())

    val vecY = sc.parallelize(Seq(5.0, 6.0))
    val y = vecY.collect()

    // Compute XTX and its inverse

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

    // Gradient Descent function

    def gradientDescent(iterations: Int, alpha: Double, initialTheta: DenseVector[Double], data: RDD[(DenseVector[Double], Double)]): (DenseVector[Double], Array[Double]) = {
      var weights = initialTheta
      var errors = Array[Double]()

      for (_ <- 1 to iterations) {
        val gradients = data.map { case (x, y) => computeSummand(weights, x, y) }
        val sumGradients = gradients.reduce((a, b) => a + b)
        weights -= sumGradients * alpha

        val predictions = data.map { case (x, y) => (y, weights.dot(x)) }
        val rmse = computeRMSE(predictions)
        errors = errors :+ rmse
      }

      (weights, errors)
    }

    // Function to compute summand

    def computeSummand(theta: DenseVector[Double], x: DenseVector[Double], y: Double): DenseVector[Double] = {
      val error = theta.dot(x) - y
      (theta - error) * x
    }

    // Function to compute RMSE

    def computeRMSE(predictions: RDD[(Double, Double)]): Double = {
      val squaredErrors = predictions.map { case (label, prediction) => math.pow(label - prediction, 2) }
      math.sqrt(squaredErrors.mean())
    }

    // Set parameters for gradient descent

    val initialTheta = DenseVector.zeros[Double](2)
    val alpha = 0.001
    val numIterations = 5

    // Run gradient descent
    
    val (finalWeights, trainingErrors) = gradientDescent(numIterations, alpha, initialTheta, trainingData)
    println(s"Final weights: $finalWeights")
    println(s"Training errors: ${trainingErrors.mkString(", ")}")
  }
}
