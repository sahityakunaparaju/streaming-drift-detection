from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("PipelineTest")
    .master("local[*]")
    .getOrCreate()
)

df = spark.createDataFrame(
    [(1, "hello world"), (2, "pipeline works")],
    ["id", "message"]
)

df.show()

spark.stop()