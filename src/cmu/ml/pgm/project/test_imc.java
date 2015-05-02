package cmu.ml.pgm.project;

/**
 * Created by dexter on 5/2/15.
 */
public class test_imc {
    public static void main(String[] args) {
        String directory = "Data/IMC/";
        DataGeneDisease data = new DataGeneDisease(directory + "gene_features.csv", directory + "disease_features.csv",
                directory + "data.csv", directory + "info");

        data.printMatrix();
    }
}
