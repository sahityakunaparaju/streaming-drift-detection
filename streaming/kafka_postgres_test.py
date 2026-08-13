from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("KafkaPostgresTest")
    .getOrCreate()
)

kafka_df = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "kafka:29092")
    .option("subscribe", "pipeline-test")
    .option("startingOffsets", "earliest")
    .load()
)

messages = kafka_df.selectExpr(
    "CAST(value AS STRING) AS message"
)

def write_to_postgres(batch_df, batch_id):
    (
        batch_df.write
        .format("jdbc")
        .option("url", "jdbc:postgresql://postgres:5432/drift_db")
        .option("dbtable", "pipeline_test")
        .option("user", "drift")
        .option("password", "drift")
        .option("driver", "org.postgresql.Driver")
        .mode("append")
        .save()
    )

query = (
    messages.writeStream
    .foreachBatch(write_to_postgres)
    .outputMode("append")
    .option("checkpointLocation", "/tmp/postgres-test-checkpoint")
    .start()
)

query.awaitTermination()