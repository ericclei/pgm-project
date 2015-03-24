package cmu.ml.pgm.project;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;

class MatrixFactorizationMovieLens {
    private DenseMatrix uFeatureMatrix;
    private DenseMatrix iFeatureMatrix;
    private LinkedSparseMatrix relationMatrix;
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

	public LinkedSparseMatrix getRelationMatrix() {
		return relationMatrix;
	}

    public ArrayList<Pair> getTrainingData() {
        return trainingData;
    }

    public int[] getNumDataPerUser() { return numDataPerUser; }

    public int[] getNumDataPerItem() { return numDataPerItem; }

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

	public MatrixFactorizationMovieLens(String userFeatureFilename, String itemFeatureFilename,
                         String relationFilename, String summaryFilename) {
        uFeatureSize = 24; // age, gender, 22 occupations (binary)
        iFeatureSize = 20; // release date, 19 genres (binary)
        try {
            BufferedReader summaryIn = new BufferedReader(new FileReader(summaryFilename));
            numUsers = Integer.parseInt(summaryIn.readLine().split(" ")[0]);
            numItems = Integer.parseInt(summaryIn.readLine().split(" ")[0]);
            summaryIn.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }

        uFeatureMatrix = new DenseMatrix(numUsers, uFeatureSize);
        initializeUserMatrix(userFeatureFilename);
        iFeatureMatrix = new DenseMatrix(numItems, iFeatureSize);
        initializeItemMatrix(itemFeatureFilename);
        relationMatrix = new LinkedSparseMatrix(numUsers, numItems);
        trainingData = new ArrayList<Pair>();
        numDataPerUser = new int[numUsers];
        numDataPerItem = new int[numItems];
        initializeRelationMatrix(relationFilename);
    }

    public void initializeUserMatrix(String filename) {
        try {
            BufferedReader fin = new BufferedReader(new FileReader(filename));
            HashMap<String, Integer> occupations = new HashMap<String, Integer>();
            int count = 2;
            double maxAge = 0;
            while(fin.ready()) {
                String[] tokens = fin.readLine().split("\\|");
                int userId = Integer.parseInt(tokens[0]) - 1;
                int age = Integer.parseInt(tokens[1]);
                uFeatureMatrix.set(userId, 0, age);
                if(maxAge < age) maxAge = age;

                uFeatureMatrix.set(userId, 1, (tokens[2].equals("M")) ? 1 : 0);

                int occupationId = count;
                if(occupations.containsKey(tokens[3])) {
                    occupationId = occupations.get(tokens[3]);
                } else {
                    occupations.put(tokens[3], count++);
                }

                for(int i = 2; i < uFeatureSize; i++) {
                    uFeatureMatrix.set(userId, i, (occupationId == i) ? 1 : 0);
                }
            }
            // Normalize age to have a value between 0 and 1
            for(int i = 0; i < uFeatureMatrix.numRows(); i++) {
                uFeatureMatrix.set(i, 0, uFeatureMatrix.get(i, 0) / maxAge);
            }
            fin.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
    }

    public void initializeItemMatrix(String filename) {
        try {
            BufferedReader fin = new BufferedReader(new FileReader(filename));
            double maxYear = 0;
            double minYear = Double.MAX_VALUE;
            while(fin.ready()) {
                String[] tokens = fin.readLine().split("\\|");
                int itemId = Integer.parseInt(tokens[0]) - 1;
                if(tokens[2].length() > 2) {
                    int year = Integer.parseInt(tokens[2].substring(tokens[2].length() - 2));
                    if (maxYear < year) maxYear = year;
                    if (minYear > year) minYear = year;
                    iFeatureMatrix.add(itemId, 0, year);
                } else {
                    continue;
                }
                for(int i = 1; i < iFeatureSize; i++) {
                    iFeatureMatrix.add(itemId, i, Integer.parseInt(tokens[i + 4]));
                }
            }

            double normalizationConstant = maxYear - minYear;
            // Normalize age to have a value between 0 and 1
            for(int i = 0; i < iFeatureMatrix.numRows(); i++) {
                iFeatureMatrix.set(i, 0, (iFeatureMatrix.get(i, 0) - minYear)/normalizationConstant);
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
            while(fin.ready()) {
                String[] tokens = fin.readLine().split("\t");
                int user_id = Integer.parseInt(tokens[0]) - 1;
                int item_id = Integer.parseInt(tokens[1]) - 1;
                int rating = Integer.parseInt(tokens[2]);
                relationMatrix.set(user_id, item_id, rating / 5.0);
                trainingData.add(new Pair(user_id, item_id, rating / 5.0));
                numDataPerUser[user_id]++;
                numDataPerItem[item_id]++;
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
}