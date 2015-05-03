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

class MatrixFactorizationMovieLens implements
		CollectiveMatrixFactorizationDataset {
	private int datasetType;
	private DenseMatrix uFeatureMatrix;
	private DenseMatrix iFeatureMatrix;
	private ArrayList<String> uFeatureType;
	private ArrayList<String> iFeatureType;
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

	public ArrayList<String> getuFeatureType() {
		return uFeatureType;
	}

	public ArrayList<String> getiFeatureType() {
		return iFeatureType;
	}

	@Override
	public Matrix getUserUserMatrix() {
		return null;
	}

	@Override
	public Matrix getItemItemMatrix() {
		return null;
	}

	public LinkedSparseMatrix getRelationMatrix() {
		return relationMatrix;
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

	public MatrixFactorizationMovieLens(String userFeatureFilename,
			String itemFeatureFilename, String relationFilename,
			String summaryFilename) {
		this(userFeatureFilename, itemFeatureFilename, relationFilename,
				summaryFilename, 0);
	}

	public MatrixFactorizationMovieLens(String userFeatureFilename,
			String itemFeatureFilename, String relationFilename,
			String summaryFilename, int dataset) {
		datasetType = dataset;
		uFeatureSize = 29; // 7 age, gender, 21 occupations (binary)
		iFeatureSize = 20; // release date, 19 genres (binary)
		try {
			BufferedReader summaryIn = new BufferedReader(new FileReader(
					summaryFilename));
			numUsers = Integer.parseInt(summaryIn.readLine().split(" ")[0]);
			numItems = Integer.parseInt(summaryIn.readLine().split(" ")[0]);
			summaryIn.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}

		uFeatureMatrix = new DenseMatrix(numUsers, uFeatureSize);
		initializeUserMatrix(userFeatureFilename);
		uFeatureType = new ArrayList<String>();
		for (int i = 0; i < uFeatureSize; i++) {
			uFeatureType.add("b");
		}

		iFeatureMatrix = new DenseMatrix(numItems, iFeatureSize);
		initializeItemMatrix(itemFeatureFilename);

		iFeatureType = new ArrayList<String>(iFeatureSize);
		iFeatureType.add("c");
		for (int i = 1; i < iFeatureSize; i++) {
			iFeatureType.add("b");
		}
		relationMatrix = new LinkedSparseMatrix(numUsers, numItems);
		trainingData = new ArrayList<Pair>();
		numDataPerUser = new int[numUsers];
		numDataPerItem = new int[numItems];
		initializeRelationMatrix(relationFilename);
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
			HashMap<String, Integer> occupations = new HashMap<String, Integer>();
			int count = 2;
			double maxAge = 0;
			while (fin.ready()) {
				String delim = "\\|";
				if (datasetType == 1)
					delim = "::";
				String[] tokens = fin.readLine().split(delim);
				int userId = Integer.parseInt(tokens[0]) - 1;
				int genderIndex = 1;
				if (datasetType == 0)
					genderIndex = 2;

				int age = discretizeAge(Integer
						.parseInt(tokens[3 - genderIndex]));
				for (int i = 0; i <= 6; i++) {
					uFeatureMatrix.set(userId, i, (age == i) ? 1 : 0);
				}
				// uFeatureMatrix.set(userId, 0, age);
				// if(maxAge < age) maxAge = age;
				uFeatureMatrix.set(userId, 7,
						(tokens[genderIndex].equals("M")) ? 1 : 0);

				int occupationId = count;
				if (occupations.containsKey(tokens[3])) {
					occupationId = occupations.get(tokens[3]);
				} else {
					occupations.put(tokens[3], count++);
				}

				for (int i = 8; i < uFeatureSize; i++) {
					uFeatureMatrix.set(userId, i, (occupationId == i) ? 1 : 0);
				}
			}
			// Normalize age to have a value between 0 and 1
			// for(int i = 0; i < uFeatureMatrix.numRows(); i++) {
			// uFeatureMatrix.set(i, 0, uFeatureMatrix.get(i, 0) / maxAge);
			// }
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
			String[] genres = { "unknown", "Action", "Adventure", "Animation",
					"Children's", "Comedy", "Crime", "Documentary", "Drama",
					"Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
					"Romance", "Sci-Fi", "Thriller", "War", "Western" };
			double maxYear = 0;
			double minYear = Double.MAX_VALUE;
			while (fin.ready()) {
				String delim = "\\|";
				if (datasetType == 1)
					delim = "::";
				String[] tokens = fin.readLine().split(delim);
				int itemId = Integer.parseInt(tokens[0]) - 1;
				int year = 0;
				if (datasetType == 0) {
					if (tokens[2].length() > 2) {
						year = Integer.parseInt(tokens[2].substring(tokens[2]
								.length() - 2));
					} else {
						continue;
					}
				} else if (datasetType == 1) {
					year = Integer.parseInt(tokens[1].substring(
							tokens[1].length() - 5, tokens[1].length() - 1));
				}
				if (maxYear < year)
					maxYear = year;
				if (minYear > year)
					minYear = year;
				iFeatureMatrix.add(itemId, 0, year);
				if (datasetType == 0) {
					for (int i = 1; i < iFeatureSize; i++) {
						iFeatureMatrix.add(itemId, i,
								Integer.parseInt(tokens[i + 4]));
					}
				} else if (datasetType == 1) {
					HashSet<String> current_genres = new HashSet<String>(
							Arrays.asList(tokens[2].split("\\|")));
					int count = 1;
					for (String genre : genres) {
						if (current_genres.contains(genre)) {
							iFeatureMatrix.add(itemId, count, 1);
						} else {
							iFeatureMatrix.add(itemId, count, 0);
						}
					}
				}
			}

			double normalizationConstant = maxYear - minYear;
			// Normalize age to have a value between 0 and 1
			for (int i = 0; i < iFeatureMatrix.numRows(); i++) {
				if (iFeatureMatrix.get(i, 0) == 0)
					iFeatureMatrix.set(i, 0, .5);
				else
					iFeatureMatrix.set(i, 0, (iFeatureMatrix.get(i, 0) - minYear)
							/ normalizationConstant);
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
				String delim = "\t";
				if (datasetType == 1)
					delim = "::";
				String[] tokens = fin.readLine().split(delim);
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