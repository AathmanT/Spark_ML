package binod.suman.SparkML;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

// $example on$
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class KMeansExample {
    public static void main(String[] args) {

        KMeansExample demo = new KMeansExample();
        JavaSparkContext jsc = demo.createSparkContext();

//        SparkConf conf = new SparkConf().setAppName("JavaKMeansExample");
//        JavaSparkContext jsc = new JavaSparkContext(conf);

        // $example on$
        // Load and parse data
//        String path = "kmeans_data.txt";
//        JavaRDD<String> data = jsc.textFile(path);
//        JavaRDD<Vector> parsedData = data.map(s -> {
//            String[] sarray = s.split(" ");
//            double[] values = new double[sarray.length];
//            for (int i = 0; i < sarray.length; i++) {
//                values[i] = Double.parseDouble(sarray[i]);
//            }
//            return Vectors.dense(values);
//        });
//        parsedData.cache();

        List<Vector> vlist = new ArrayList<Vector>();
        for (int i=0;i<9;i++){
            vlist.add(Vectors.dense(i));
        }
        System.out.println(Arrays.toString(vlist.toArray()));

        JavaRDD<Vector> data = jsc.parallelize(vlist);

        double[] testData = new double[] { 1.0, 1.0, 9839.64, 170136.0, 160296.36, 0.0, 0.0};
        Vector newData = Vectors.dense(testData);

        Double[] tData = new Double[] { 1.0, 20.0, 366.0, 9839.64, 20.0, 170136.0, 160296.36,23.0,246.0};
        List<Double> tdata = Arrays.asList(tData);

        // note that each Vector is a row and not a column
        JavaRDD<Vector> data1 = jsc.parallelize(
                Arrays.asList(
                        Vectors.dense(1.0),
                        Vectors.dense(20.0),
                        Vectors.dense(366.0),
                        Vectors.dense(9839.64),
                        Vectors.dense(20.0),
                        Vectors.dense(170136.0),
                        Vectors.dense(160296.36),
                        Vectors.dense(23.0),
                        Vectors.dense(246.0)
                )
        );

        // Cluster the data into two classes using KMeans
        int numClusters = 2;
        int numIterations = 20;
        KMeansModel kMeansModel = KMeans.train(data.rdd(), numClusters, numIterations);

        System.out.println("Cluster centers:");
        for (Vector center: kMeansModel.clusterCenters()) {
            System.out.println(" " + center);
        }

        int pred = kMeansModel.predict(Vectors.dense(20));
        System.out.println("The pred: "+pred);


        JavaRDD<Integer> cluster_ind = kMeansModel.predict(jsc.parallelize(data.collect()));
        List<Integer> cluster_inds = cluster_ind.collect();
        System.out.println(Arrays.toString(cluster_inds.toArray()));

        int search_key = 0;
        // Collect matches
        List<Integer> matchingIndices = new ArrayList<>();
        for (int i = 0; i < cluster_inds.size(); i++) {
            int element = cluster_inds.get(i);

            if (search_key == element) {
                matchingIndices.add(i);
            }
        }
        System.out.println(Arrays.toString(matchingIndices.toArray()));

        List<Double> filtered = matchingIndices.stream()
                .map(tdata::get)
                .collect(Collectors.toList());

        System.out.println(Arrays.toString(filtered.toArray()));

        double cost = kMeansModel.computeCost(data.rdd());
        System.out.println("Cost: " + cost);

        // Evaluate clustering by computing Within Set Sum of Squared Errors
        double WSSSE = kMeansModel.computeCost(data.rdd());
        System.out.println("Within Set Sum of Squared Errors = " + WSSSE);

//        // Save and load model
//        clusters.save(jsc.sc(), "target/org/apache/spark/JavaKMeansExample/KMeansModel");
//        KMeansModel sameModel = KMeansModel.load(jsc.sc(),
//                "target/org/apache/spark/JavaKMeansExample/KMeansModel");
//        // $example off$

        jsc.stop();
    }

    public JavaSparkContext createSparkContext() {
        SparkConf conf = new SparkConf().setAppName("Main")
                .setMaster("local[2]")
                .set("spark.executor.memory", "3g")
                .set("spark.driver.memory", "3g");

        JavaSparkContext sc = new JavaSparkContext(conf);
        return sc;
    }
}
