package cmu.ml.pgm.project;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;

import java.util.ArrayList;

public interface CollectiveMatrixFactorizationDataset {

	int getNumEntities();
	
	Matrix getRelations(int s, int t);
	
	Matrix getBernoulliFeatures(int s);

	Matrix getNormalFeatures(int s);
	
	int getNumItems(int s);
	
	int getNumBernoulliFeatures(int s);

	int getNumNormalFeatures(int s);
	
	int getNumObserved(int s, int t);

	Matrix getRelationMatrix();

	DenseMatrix getuFeatureMatrix();

	DenseMatrix getiFeatureMatrix();

	ArrayList<String> getuFeatureType();

	ArrayList<String> getiFeatureType();
}
