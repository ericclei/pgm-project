package cmu.ml.pgm.project;

import static cmu.ml.pgm.project.MatrixMethods.*;

import java.util.ArrayList;
import java.util.List;

import no.uib.cipr.matrix.Matrix;

public class CollectiveMatrixFactorizationResult {

	// private Matrix[][] relations;
	private Matrix[] latentFeatures;
	private Matrix[] bernoulliFeatureMaps;
	private Matrix[] normalFeatureMaps;
	private double[][] relationVariances;
	private double[] featureVariances;
	private List<Matrix[][]> intermediateRelations;

	public CollectiveMatrixFactorizationResult(int numEntities) {
		latentFeatures = new Matrix[numEntities];
		bernoulliFeatureMaps = new Matrix[numEntities];
		normalFeatureMaps = new Matrix[numEntities];
		relationVariances = new double[numEntities][numEntities];
		featureVariances = new double[numEntities];
		intermediateRelations = new ArrayList<Matrix[][]>();
	}

	public int getNumEntities() {
		return latentFeatures.length;
	}

	/**
	 * Warning: computes R on each call.
	 * 
	 * @param s
	 * @param t
	 * @return
	 */
	public Matrix getRelations(int s, int t) {
		return times(latentFeatures[s], transpose(latentFeatures[t]));
	}

	public Matrix getLatentFeatures(int s) {
		return latentFeatures[s];
	}

	public void setLatentFeatures(int s, Matrix r) {
		latentFeatures[s] = r;
	}

	public Matrix getBernoulliFeatureMap(int s) {
		return bernoulliFeatureMaps[s];
	}

	public void setBernoulliFeatureMap(int s, Matrix m) {
		bernoulliFeatureMaps[s] = m;
	}

	public Matrix getNormalFeatureMap(int s) {
		return normalFeatureMaps[s];
	}

	public void setNormalFeatureMap(int s, Matrix m) {
		normalFeatureMaps[s] = m;
	}

	public double getRelationVariance(int s, int t) {
		return relationVariances[s][t];
	}

	public void setRelationVariance(int s, int t, double v) {
		relationVariances[s][t] = v;
		relationVariances[t][s] = v;
	}

	public double getFeatureVariance(int s) {
		return featureVariances[s];
	}

	public void setFeatureVariance(int s, double v) {
		featureVariances[s] = v;
	}

	public Matrix getIntermediateRelations(int i, int s, int t) {
		assert i < getNumIntermediate();
		if (s <= t)
			return intermediateRelations.get(i)[s][t];
		return transpose(intermediateRelations.get(i)[t][s]);
	}

	public int getNumIntermediate() {
		return intermediateRelations.size();
	}

	public void addIntermediateRelations() {
		int nE = getNumEntities();
		Matrix[][] rs = new Matrix[nE][nE];
		for (int s = 0; s < nE; s++) {
			for (int t = s; t < nE; t++) {
				rs[s][t] = times(latentFeatures[s],
						transpose(latentFeatures[t]));
			}
		}
		intermediateRelations.add(rs);
	}
}
