package cmu.ml.pgm.project;


import gov.sandia.cognition.math.matrix.mtj.DenseMatrix;
import gov.sandia.cognition.math.matrix.mtj.DenseMatrixFactoryMTJ;
import gov.sandia.cognition.math.matrix.mtj.DenseVector;
import gov.sandia.cognition.math.matrix.mtj.DenseVectorFactoryMTJ;
import gov.sandia.cognition.statistics.distribution.InverseWishartDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;

import java.util.Random;

import no.uib.cipr.matrix.Matrices;
import no.uib.cipr.matrix.Matrix;


/**
 * Methods for feature-enriched matrix factorization. Static class.
 * @author dexter
 *
 */
public final class BayesianMatrixFactorization {

    private static final Random rand = new Random(1);
    private static final DenseMatrixFactoryMTJ mfactory = new DenseMatrixFactoryMTJ();
	private static final DenseVectorFactoryMTJ vfactory = new DenseVectorFactoryMTJ(); 
	
	private static class Hyperparams {
		DenseVector mu;
		DenseMatrix cov;
		
		Hyperparams(gov.sandia.cognition.math.matrix.Vector mu_i, 
				gov.sandia.cognition.math.matrix.Matrix cov_i) {
			mu = (DenseVector) mu_i.clone();
			cov = (DenseMatrix) cov_i.clone();
		}
	}
	
    private BayesianMatrixFactorization() {}

    /**
     *
     * @param data
     * @param latentDim
     * @param numSamples
     * @param alpha : alpha_u, alpha_v, alpha_a, alpha_b
     * @return estimates (R, Us, Vs, As, Bs)
     */
    public static MatrixFactorizationResult factorizeMatrixWithFeatures(MatrixFactorizationMovieLens data,
                                                                        int latentDim, int numSamples, double alpha) {
		Matrix R = data.getRelationMatrix();
		Matrix F = data.getuFeatureMatrix();
		Matrix G = data.getiFeatureMatrix();
		int dF = F.numColumns();
		int dG = G.numColumns();
		int m = F.numRows();
		int n = G.numRows();
		int numBuffer = 0;
		int length = numSamples + numBuffer + 1;
		Matrix[] U = new Matrix[length];
		U[0] = new no.uib.cipr.matrix.DenseMatrix(m, latentDim);
		Matrix[] V = new Matrix[length];
		V[0] = new no.uib.cipr.matrix.DenseMatrix(n, latentDim);
		Matrix[] A = new Matrix[length];
		A[0] = new no.uib.cipr.matrix.DenseMatrix(dF, latentDim);
		Matrix[] B = new Matrix[length];
		B[0] = new no.uib.cipr.matrix.DenseMatrix(dG, latentDim);
		randomlyInitialize(U[0]);
		randomlyInitialize(V[0]);
		randomlyInitialize(A[0]);
		randomlyInitialize(B[0]);

		Hyperparams[] hpU = new Hyperparams[length];
		Hyperparams[] hpV = new Hyperparams[length];
		Hyperparams[] hpA = new Hyperparams[length];
		Hyperparams[] hpB = new Hyperparams[length];

//		numSamples = 1;
		for(int t = 0; t < numSamples + numBuffer; t++) {
//			System.out.println("Sample: " + (t + 1));
			//Sample Hyperparams - can be done in parallel
			sampleHyperparams(U[t], hpU, t);
			sampleHyperparams(V[t], hpV, t);
			sampleHyperparams(A[t], hpA, t);
			sampleHyperparams(B[t], hpB, t);
			//Sample latent variables
			sampleLatentVariables(hpA[t], A, t + 1, F, U[t], null, null, alpha);
			sampleLatentVariables(hpB[t], B, t + 1, G, V[t], null, null, alpha);
			sampleLatentVariables(hpU[t], U, t + 1, F, A[t + 1], R, V[t], alpha);
			sampleLatentVariables(hpV[t], V, t + 1, G, B[t + 1], transpose(R), U[t + 1], alpha);
		}
		
		Matrix result = new no.uib.cipr.matrix.DenseMatrix(m, n);
		for(int i = numBuffer + 1; i < U.length; i++) {
			result.add(matrixMult(U[i], transpose(V[i])));
		}
        return new MatrixFactorizationResult(result.scale(1.0/numSamples), U, V);
    }
    
    static void sampleHyperparams(Matrix Fin, Hyperparams[] hp, int t) {
    	DenseMatrix F = mfactory.copyArray(Matrices.getArray(Fin));
    	DenseVector Fbar = average(F); //Sample mean
    	DenseMatrix S = covariance(F, Fbar); //Sample covariance
    	int latentDim = F.getNumColumns();
    	int n = F.getNumRows();
    	
    	DenseMatrix Psi = (DenseMatrix) S.plus(mfactory.createIdentity(latentDim, latentDim)).plus(Fbar.outerProduct(Fbar).scale(n/(n + 1.0)));
    	InverseWishartDistribution iw = new InverseWishartDistribution(Psi, latentDim + n);
		DenseMatrix cov = (DenseMatrix) iw.sample(rand); //Sample covariance matrix from Wishart distribution
		MultivariateGaussian normal = new MultivariateGaussian(Fbar.scale(n / (1.0 + n)), cov.scale(1.0/(1 + n)));
		hp[t] = new Hyperparams(normal.sample(rand), cov); //Sample mean from Multivariate Gaussian distribution
    }
    
    static void sampleLatentVariables(Hyperparams hpin, Matrix[] F, int t, Matrix RF, Matrix Ain, Matrix R, Matrix Ein, double alpha) {
    	F[t] = new no.uib.cipr.matrix.DenseMatrix(F[t-1].numRows(), F[t-1].numColumns());
    	DenseMatrix A = mfactory.copyArray(Matrices.getArray(Ain));
    	DenseMatrix E = null;
    	if(Ein != null) E = mfactory.copyArray(Matrices.getArray(Ein));
    	int latentDim = F[t].numColumns();
    	DenseVector precTimesMu = (DenseVector) hpin.cov.inverse().times(hpin.mu);

    	DenseMatrix prec = mfactory.createMatrix(latentDim, latentDim);
    	for(int i = 0; i < A.getNumRows(); i++) {
    		prec.plusEquals(A.getRow(i).outerProduct(A.getRow(i)));
    	}
    	prec.scaleEquals(alpha);
    	prec.plusEquals(hpin.cov.inverse());
    	
    	for(int i = 0; i < F[t].numRows(); i++) {
    		DenseMatrix prec_ind = prec.clone();
    		DenseVector mu = (DenseVector) precTimesMu.clone();
    		for(int j = 0; j < A.getNumRows(); j++) {
    			double rf;
    			if(E == null) {
    				rf = RF.get(j, i);
    			} else {
    				rf = RF.get(i, j);
    			}
    			mu.plusEquals(A.getRow(j).scale(alpha * rf));
    		}
    		if(E != null) {
    			for(int j = 0; j < E.getNumRows(); j++) {
    				if(R.get(i, j) > 0) {
    					prec_ind.plusEquals(E.getRow(j).outerProduct(E.getRow(j)).scale(alpha));
    					mu.plusEquals(E.getRow(j).scale(alpha * R.get(i, j)));
    				}
    			}
    		}
    		DenseMatrix cov_ind = (DenseMatrix) prec_ind.inverse();
    		mu = (DenseVector) mu.times(cov_ind);
    		MultivariateGaussian normal = new MultivariateGaussian(mu, cov_ind);
    		setRow(F[t], (DenseVector) normal.sample(rand), i);
    	}
    }
    
    static DenseVector average(DenseMatrix x) {
    	DenseVector result = vfactory.createVector(x.getNumColumns());
    	for (int i = 0; i < x.getNumRows(); i++) {
    		result.plusEquals(x.getRow(i));
    	}
    	result.scaleEquals(1.0/x.getNumRows());
    	return result;
    }

    static DenseMatrix covariance(DenseMatrix x, DenseVector mu) {
    	DenseMatrix result = mfactory.createMatrix(x.getNumColumns(), x.getNumColumns());
    	for (int i = 0; i < x.getNumRows(); i++) {
    		DenseVector diff = (DenseVector) x.getRow(i).minus(mu);
    		result.plusEquals(diff.outerProduct(diff));
    	}
    	return result;
    }
    
    static Matrix matrixMult(Matrix x, Matrix y) {
        return x.mult(y, new no.uib.cipr.matrix.DenseMatrix(x.numRows(), y.numColumns()));
    }
    
    static void setRow(Matrix x, DenseVector v, int index) {
    	double[] update = v.toArray();
    	for(int i = 0; i < update.length; i++) {
    		x.set(index, i, update[i]);
    	}
    }

    /**
     * U(-1, 1)
     * @param x
     */
    static void randomlyInitialize(Matrix x) {
        Random r = new Random();
        for (int i = 0; i < x.numRows(); i++)
            for (int j = 0; j < x.numColumns(); j++)
                x.set(i, j, 2 * r.nextDouble() - 1);
    }

    static void setAllValues(Matrix x, double val) {
        for (int i = 0; i < x.numRows(); i++)
            for (int j = 0; j < x.numColumns(); j++)
                x.set(i, j, val);
    }

    static Matrix transpose(Matrix x) {
        return x.transpose(new no.uib.cipr.matrix.DenseMatrix(x.numColumns(), x.numRows()));
    }
}