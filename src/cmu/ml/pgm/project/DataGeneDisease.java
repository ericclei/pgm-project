package cmu.ml.pgm.project;

import static cmu.ml.pgm.project.MatrixMethods.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;

public class DataGeneDisease implements CollectiveMatrixFactorizationDataset {
	private DenseMatrix uFeatureMatrix;
	private DenseMatrix iFeatureMatrix;
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

	public DataGeneDisease(String directory, boolean isTraining,
			boolean useFeatures) {
		try {
			BufferedReader summaryIn = new BufferedReader(new FileReader(
					directory + "info"));
			numUsers = Integer.parseInt(summaryIn.readLine().split(" ")[0]);
			numItems = Integer.parseInt(summaryIn.readLine().split(" ")[0]);
			if (useFeatures) {
				uFeatureSize = Integer
						.parseInt(summaryIn.readLine().split(" ")[0]);
				iFeatureSize = Integer
						.parseInt(summaryIn.readLine().split(" ")[0]);
			} else {
				uFeatureSize = iFeatureSize = 0;
			}
			summaryIn.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}

		uFeatureMatrix = new DenseMatrix(numUsers, uFeatureSize);
		iFeatureMatrix = new DenseMatrix(numItems, iFeatureSize);
		if (useFeatures) {
			initializeUserMatrix(directory + "gene_features.csv");
			initializeItemMatrix(directory + "disease_features.csv");
		}

		relationMatrix = new LinkedSparseMatrix(numUsers, numItems);
		trainingData = new ArrayList<Pair>();
		numDataPerUser = new int[numUsers];
		numDataPerItem = new int[numItems];

		String dataFile = isTraining ? "data_train.csv" : "data_test.csv";
		initializeRelationMatrix(directory + dataFile);

		String uuFile = isTraining ? "gene_gene_train.csv"
				: "gene_gene_test.csv";
		uuMatrix = new LinkedSparseMatrix(numUsers, numUsers);
		initializeUserUserMatrix(directory + uuFile);

		String iiFile = isTraining ? "disease_disease.csv"
				: "disease_disease_test.csv";
		iiMatrix = new DenseMatrix(numUsers, numItems);
		initializeItemItemMatrix(directory + iiFile);
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
				for (int i = 0; i < tokens.length; i++) {
					uFeatureMatrix
							.set(userId, i, Double.parseDouble(tokens[i]));
				}
				userId++;
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
				for (int i = 0; i < tokens.length; i++) {
					iFeatureMatrix
							.set(itemId, i, Double.parseDouble(tokens[i]));
				}
				itemId++;
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
				int user_id = Integer.parseInt(tokens[0]);
				int item_id = Integer.parseInt(tokens[1]);
				double rating = Integer.parseInt(tokens[2]);
				rating = (rating + 1.0) / 2;
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
			while (fin.ready()) {
				String delim = ",";
				String[] tokens = fin.readLine().split(delim);
				int user_id = Integer.parseInt(tokens[0]) - 1;
				int item_id = Integer.parseInt(tokens[1]) - 1;
				double rating = Double.parseDouble(tokens[2]);
				iiMatrix.set(user_id, item_id, rating);
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
		if (s == 0) {
			if (t == 0) {
				return uuMatrix;
			} else {
				return relationMatrix;
			}
		} else {
			if (t == 0) {
				return transpose(relationMatrix);
			} else {
				return iiMatrix;
			}
		}
	}

	@Override
	public Matrix getBernoulliFeatures(int s) {
		return new DenseMatrix(numUsers, 0);
	}

	@Override
	public Matrix getNormalFeatures(int s) {
		if (s == 0)
			return uFeatureMatrix;
		else
			return iFeatureMatrix;
	}

	@Override
	public Matrix getFeatures(int s) {
		return getNormalFeatures(s);
	}

	@Override
	public List<String> getFeatureTypes(int s) {
		List<String> result = new ArrayList<String>(getNumNormalFeatures(s));
		for(int i = 0; i < getNumNormalFeatures(s); i++) {
			result.add("c");
		}
		return result;
	}

	@Override
	public int getNumItems(int s) {
		return s == 0 ? numUsers : numItems;
	}

	@Override
	public int getNumBernoulliFeatures(int s) {
		return 0;
	}

	@Override
	public int getNumNormalFeatures(int s) {
		return s == 0 ? getuFeatureSize() : getiFeatureSize();
	}

	@Override
	public int getNumObserved(int s, int t) {
		if (s == 0) {
			if (t == 0) {
				return l0Norm(uuMatrix);
			} else {
				return l0Norm(relationMatrix);
			}
		} else {
			if (t == 0) {
				return l0Norm(relationMatrix);
			} else {
				return l0Norm(iiMatrix);
			}
		}
	}
}
