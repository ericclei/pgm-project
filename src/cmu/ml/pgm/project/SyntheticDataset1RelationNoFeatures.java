package cmu.ml.pgm.project;

import java.io.BufferedReader;
import java.io.FileReader;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;
import static cmu.ml.pgm.project.MatrixMethods.*;

/**
 * When indexing entities of this dataset, use 0, 1, 2.
 * 
 * @author eric
 *
 */
public class SyntheticDataset1RelationNoFeatures implements
		CollectiveMatrixFactorizationDataset {

	private Matrix f1, f2, r12;
	private final int N_ENTITIES = 2;
	private final int[] N = { 1000, 1000 };
	private final int[] D = { 0, 0 };
	private final int N_OBS_R12 = 50000;

	public SyntheticDataset1RelationNoFeatures(String r12Path, boolean isSparse) {
		if (isSparse) {
			r12 = new LinkedSparseMatrix(N[0], N[1]);
			initializeSparseMatrix(r12Path, r12);
		} else {
			r12 = new DenseMatrix(N[0], N[1]);
			initializeDenseMatrix(r12Path, r12);
		}

		f1 = new DenseMatrix(N[0], D[0]);
		f2 = new DenseMatrix(N[1], D[1]);
	}

	void initializeDenseMatrix(String filename, Matrix r) {
		try {
			BufferedReader fin = new BufferedReader(new FileReader(filename));
			int i = 0;
			while (fin.ready()) {
				String delim = ",";
				String[] tokens = fin.readLine().split(delim);
				for (int j = 0; j < tokens.length; j++) {
					float val = Float.parseFloat(tokens[j]);
					r.set(i, j, val);
				}
				i++;
			}
			fin.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

	void initializeSparseMatrix(String filename, Matrix r) {
		try {
			BufferedReader fin = new BufferedReader(new FileReader(filename));
			while (fin.ready()) {
				String delim = ",";
				String[] tokens = fin.readLine().split(delim);
				int user_id = Integer.parseInt(tokens[0]) - 1;
				int item_id = Integer.parseInt(tokens[1]) - 1;
				float rating = Float.parseFloat(tokens[2]);
				r.set(user_id, item_id, rating);
			}
			fin.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

	@Override
	public int getNumEntities() {
		return N_ENTITIES;
	}

	@Override
	public Matrix getRelations(int s, int t) {
		assert s != t;
		assert 0 <= s && s <= 1;
		assert 0 <= t && t <= 1;
		if (s == 0)
			return r12;
		return transpose(r12);
	}

	@Override
	public Matrix getBernoulliFeatures(int s) {
		assert 0 <= s && s <= 1;
		return new DenseMatrix(N[s], 0);
	}

	@Override
	public Matrix getNormalFeatures(int s) {
		assert 0 <= s && s <= 1;
		if (s == 0)
			return f1;
		return f2;
	}

	@Override
	public int getNumItems(int s) {
		assert 0 <= s && s <= 1;
		return N[s];
	}

	@Override
	public int getNumBernoulliFeatures(int s) {
		assert 0 <= s && s <= 1;
		return 0;
	}

	@Override
	public int getNumNormalFeatures(int s) {
		assert 0 <= s && s <= 1;
		return D[s];
	}

	@Override
	public int getNumObserved(int s, int t) {
		assert s != t;
		assert 0 <= s && s <= 1;
		assert 0 <= t && t <= 1;
		return N_OBS_R12;
	}

}
