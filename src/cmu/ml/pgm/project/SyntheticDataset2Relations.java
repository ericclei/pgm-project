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
public class SyntheticDataset2Relations implements
		CollectiveMatrixFactorizationDataset {

	private Matrix f1Normal, f2Normal, f3Normal, f1Bernou, f2Bernou, f3Bernou,
			r12, r13;
	private final int N_ENTITIES = 3;
	private final int[] N = { 1000, 1000, 1000 };
	private final int[] D_NORMAL = { 50, 50, 50 };
	private final int[] D_BERNOU = { 50, 50, 50 };
	private final int N_OBS_R12 = 50000;
	private final int N_OBS_R13 = 50000;

	/**
	 * 
	 * @param f1NormalPath
	 * @param f2NormalPath
	 * @param f3NormalPath
	 * @param f1BernouPath
	 * @param f2BernouPath
	 * @param f3BernouPath
	 * @param r12Path
	 * @param r13Path
	 * @param isSparse
	 *            whether relations are sparse
	 */
	public SyntheticDataset2Relations(String f1NormalPath, String f2NormalPath,
			String f3NormalPath, String f1BernouPath, String f2BernouPath,
			String f3BernouPath, String r12Path, String r13Path,
			boolean isSparse) {
		if (isSparse) {
			r12 = new LinkedSparseMatrix(N[0], N[1]);
			r13 = new LinkedSparseMatrix(N[0], N[2]);
			initializeSparseMatrix(r12Path, r12);
			initializeSparseMatrix(r13Path, r13);
		} else {
			r12 = new DenseMatrix(N[0], N[1]);
			r13 = new DenseMatrix(N[0], N[2]);
			initializeDenseMatrix(r12Path, r12);
			initializeDenseMatrix(r13Path, r13);
		}

		f1Normal = new DenseMatrix(N[0], D_NORMAL[0]);
		f2Normal = new DenseMatrix(N[1], D_NORMAL[1]);
		f3Normal = new DenseMatrix(N[2], D_NORMAL[2]);
		initializeDenseMatrix(f1NormalPath, f1Normal);
		initializeDenseMatrix(f2NormalPath, f2Normal);
		initializeDenseMatrix(f3NormalPath, f3Normal);
		f1Bernou = new LinkedSparseMatrix(N[0], D_BERNOU[0]);
		f2Bernou = new LinkedSparseMatrix(N[1], D_BERNOU[1]);
		f3Bernou = new LinkedSparseMatrix(N[2], D_BERNOU[2]);
		initializeSparseMatrix(f1BernouPath, f1Bernou);
		initializeSparseMatrix(f2BernouPath, f2Bernou);
		initializeSparseMatrix(f3BernouPath, f3Bernou);
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
		if (s == t)
			return null;
		assert 0 <= s && s <= 2;
		assert 0 <= t && t <= 2;
		if (s == 0) {
			if (t == 1)
				return r12;
			else
				return r13;
		}

		if (s == 1) {
			if (t == 0)
				return transpose(r12);
		}

		// s == 2
		if (t == 0)
			return transpose(r13);
		return null;
	}

	@Override
	public Matrix getBernoulliFeatures(int s) {
		assert 0 <= s && s <= 2;
		if (s == 0)
			return f1Bernou;
		if (s == 1)
			return f2Bernou;
		return f3Bernou;
	}

	@Override
	public Matrix getNormalFeatures(int s) {
		assert 0 <= s && s <= 2;
		if (s == 0)
			return f1Normal;
		if (s == 1)
			return f2Normal;
		return f3Normal;
	}

	@Override
	public int getNumItems(int s) {
		assert 0 <= s && s <= 2;
		return N[s];
	}

	@Override
	public int getNumBernoulliFeatures(int s) {
		assert 0 <= s && s <= 2;
		return D_BERNOU[s];
	}

	@Override
	public int getNumNormalFeatures(int s) {
		assert 0 <= s && s <= 2;
		return D_NORMAL[s];
	}

	@Override
	public int getNumObserved(int s, int t) {
		if (s == t)
			return 0;
		assert 0 <= s && s <= 2;
		assert 0 <= t && t <= 2;
		if (s == 0) {
			if (t == 1)
				return N_OBS_R12;
			else
				return N_OBS_R13;
		}

		if (s == 1) {
			if (t == 0)
				return N_OBS_R12;
		}

		// s == 2
		if (t == 0)
			return N_OBS_R13;
		return 0;
	}

}
