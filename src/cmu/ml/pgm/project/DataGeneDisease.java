package cmu.ml.pgm.project;

import static cmu.ml.pgm.project.MatrixMethods.*;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

class DataGeneDisease implements
        CollectiveMatrixFactorizationDataset {
    private int datasetType;
    private DenseMatrix uFeatureMatrix;
    private DenseMatrix iFeatureMatrix;
    private ArrayList<String> uFeatureType;
    private ArrayList<String> iFeatureType;
    private LinkedSparseMatrix relationMatrix;
    private LinkedSparseMatrix uuMatrix;
    private DenseMatrix iiMatrix;
    private ArrayList<Pair> trainingData;
    private int[] numDataPerUser;
    private int[] numDataPerItem;
    private int numUsers;
    private int numItems;
    private int uFeatureSize;
    private int iFeatureSize;

    public class Pair {
        int user_id;
        int item_id;
        double rating;

        public Pair(int uid, int iid, double rat) {
            user_id = uid;
            item_id = iid;
            rating = rat;
        }
    }

    public DenseMatrix getuFeatureMatrix() {
        return uFeatureMatrix;
    }

    public DenseMatrix getiFeatureMatrix() {
        return iFeatureMatrix;
    }

    public ArrayList<String> getuFeatureType() {
        return uFeatureType;
    }

    public ArrayList<String> getiFeatureType() {
        return iFeatureType;
    }

    public LinkedSparseMatrix getRelationMatrix() {
        return relationMatrix;
    }

    public LinkedSparseMatrix getUserUserMatrix() {
        return uuMatrix;
    }

    public DenseMatrix getItemItemMatrix() {
        return iiMatrix;
    }

    public ArrayList<Pair> getTrainingData() {
        return trainingData;
    }

    public int[] getNumDataPerUser() {
        return numDataPerUser;
    }

    public int[] getNumDataPerItem() {
        return numDataPerItem;
    }

    public int getNumUsers() {
        return numUsers;
    }

    public int getNumItems() {
        return numItems;
    }

    public int getuFeatureSize() {
        return uFeatureSize;
    }

    public int getiFeatureSize() {
        return iFeatureSize;
    }

    public DataGeneDisease(String directory) {
        try {
            BufferedReader summaryIn = new BufferedReader(new FileReader(
                    directory + "info"));
            numUsers = Integer.parseInt(summaryIn.readLine().split(" ")[0]);
            numItems = Integer.parseInt(summaryIn.readLine().split(" ")[0]);
            uFeatureSize = Integer.parseInt(summaryIn.readLine().split(" ")[0]);
            iFeatureSize = Integer.parseInt(summaryIn.readLine().split(" ")[0]);
            summaryIn.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }

        uFeatureMatrix = new DenseMatrix(numUsers, uFeatureSize);
        initializeUserMatrix(directory + "gene_features.csv");
        uFeatureType = new ArrayList<String>();
        for (int i = 0; i < uFeatureSize; i++) {
            uFeatureType.add("c");
        }

        iFeatureMatrix = new DenseMatrix(numItems, iFeatureSize);
        initializeItemMatrix(directory + "disease_features.csv");

        iFeatureType = new ArrayList<String>(iFeatureSize);
        for (int i = 0; i < iFeatureSize; i++) {
            iFeatureType.add("c");
        }
        relationMatrix = new LinkedSparseMatrix(numUsers, numItems);
        trainingData = new ArrayList<Pair>();
        numDataPerUser = new int[numUsers];
        numDataPerItem = new int[numItems];
        initializeRelationMatrix(directory + "data.csv");

        uuMatrix = new LinkedSparseMatrix(numUsers, numUsers);
        initializeUserUserMatrix(directory + "gene_gene.csv");
    }

    /**
     * 0: age | 1: gender==M | 2-end: occupation indicator
     *
     * @param filename
     *            1: "Under 18" 18: "18-24" 25: "25-34" 35: "35-44" 45: "45-49"
     *            50: "50-55" 56: "56+"
     */
    public void initializeUserMatrix(String filename) {
        try {
            BufferedReader fin = new BufferedReader(new FileReader(filename));
            int userId = 0;
            while (fin.ready()) {
                String delim = ",";
                String[] tokens = fin.readLine().split(delim);
                for(int i = 0; i < tokens.length; i++) {
                    uFeatureMatrix.set(userId, i, Double.parseDouble(tokens[i]));
                }
                userId ++;
            }
            fin.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
    }

    public int discretizeAge(int age) {
        int[] boundary = { 18, 25, 35, 45, 50, 56 };
        for (int i = 0; i < boundary.length; i++) {
            if (age < boundary[i]) {
                return i + 1;
            }
        }
        return boundary.length;
    }

    /**
     * 0: year | 1-end: genre indicator
     *
     * @param filename
     */
    public void initializeItemMatrix(String filename) {
        try {
            BufferedReader fin = new BufferedReader(new FileReader(filename));
            int itemId = 0;
            while (fin.ready()) {
                String delim = ",";
                String[] tokens = fin.readLine().split(delim);
                for(int i = 0; i < tokens.length; i++) {
                    iFeatureMatrix.set(itemId, i, Double.parseDouble(tokens[i]));
                }
                itemId ++;
            }
            fin.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
    }

    public void initializeRelationMatrix(String filename) {
        try {
            BufferedReader fin = new BufferedReader(new FileReader(filename));
            while (fin.ready()) {
                String delim = ",";
                String[] tokens = fin.readLine().split(delim);
                int user_id = Integer.parseInt(tokens[0]) - 1;
                int item_id = Integer.parseInt(tokens[1]) - 1;
                int rating = Integer.parseInt(tokens[2]);
                relationMatrix.set(user_id, item_id, rating);
                trainingData.add(new Pair(user_id, item_id, rating));
                numDataPerUser[user_id]++;
                numDataPerItem[item_id]++;
            }
            fin.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
    }

    public void initializeUserUserMatrix(String filename) {
        try {
            BufferedReader fin = new BufferedReader(new FileReader(filename));
            while (fin.ready()) {
                String delim = ",";
                String[] tokens = fin.readLine().split(delim);
                int user_id = Integer.parseInt(tokens[0]) - 1;
                int item_id = Integer.parseInt(tokens[1]) - 1;
                double rating = Double.parseDouble(tokens[2]);
                uuMatrix.set(user_id, item_id, rating);
            }
            fin.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
    }

    public void initializeItemItemMatrix(String filename) {
        try {
            BufferedReader fin = new BufferedReader(new FileReader(filename));
            int itemId = 0;
            while (fin.ready()) {
                String delim = ",";
                String[] tokens = fin.readLine().split(delim);
                for(int i = 0; i < tokens.length; i++) {
                    iFeatureMatrix.set(itemId, i, Double.parseDouble(tokens[i]));
                }
                itemId ++;
            }
            fin.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
    }

    public void printMatrix() {
        System.out.println(relationMatrix);
    }

	/* below are interface methods */

    @Override
    public int getNumEntities() {
        return 2;
    }

    @Override
    public Matrix getRelations(int s, int t) {
        return s == 0 ? relationMatrix : transpose(relationMatrix);
    }

    @Override
    public Matrix getBernoulliFeatures(int s) {
        if (s == 0)
            return uFeatureMatrix;
        else
            return new DenseMatrix(removeColumn(getiFeatureMatrix(), 0));
    }

    @Override
    public Matrix getNormalFeatures(int s) {
        if (s == 0)
            return new DenseMatrix(numUsers, 0);
        else
            return new DenseMatrix(getColumn(getiFeatureMatrix(), 0));
    }

    @Override
    public int getNumItems(int s) {
        return s == 0 ? numUsers : numItems;
    }

    @Override
    public int getNumBernoulliFeatures(int s) {
        return s == 0 ? getuFeatureSize() : getiFeatureSize() - 1;
    }

    @Override
    public int getNumNormalFeatures(int s) {
        return s == 0 ? 0 : 1;
    }

    @Override
    public int getNumObserved(int s, int t) {
        return l0Norm(relationMatrix);
    }
}