package cmu.ml.pgm.project;


import gov.sandia.cognition.math.matrix.mtj.DenseMatrix;
import gov.sandia.cognition.math.matrix.mtj.DenseMatrixFactoryMTJ;
import gov.sandia.cognition.math.matrix.mtj.DenseVector;
import gov.sandia.cognition.math.matrix.mtj.DenseVectorFactoryMTJ;
import gov.sandia.cognition.statistics.distribution.InverseWishartDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;

import java.util.ArrayList;
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
	
	private static class Statistics {
		DenseVector mu;
		DenseMatrix cov;
		
		Statistics(gov.sandia.cognition.math.matrix.Vector mu_i,
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
		Matrix R = data.getRelations(0, 1);
		Matrix F = data.getuFeatureMatrix();
		Matrix G = data.getiFeatureMatrix();
		int dF = F.numColumns();
		int dG = G.numColumns();
		int m = F.numRows();
		int n = G.numRows();
		int numBuffer = 10;
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

		Statistics[] hpU = new Statistics[length];
		Statistics[] hpV = new Statistics[length];
		Statistics[] hpA = new Statistics[length];
		Statistics[] hpB = new Statistics[length];

		Statistics[] psU = null, psV = null, psA = null, psB = null;

//		numSamples = 1;
		for(int t = 0; t < numSamples + numBuffer; t++) {
			System.out.println("Sample: " + (t + 1));
			//Sample HyperparStatisticse done in parallel
			sampleHyperparams(U[t], hpU, t);
			sampleHyperparams(V[t], hpV, t);
			sampleHyperparams(A[t], hpA, t);
			sampleHyperparams(B[t], hpB, t);

			if(t == 0) {
				psA = initializePosteriorStatistics(hpA[t], A[t], F, U[t], null, null, alpha, data.getuFeatureType());
				psB = initializePosteriorStatistics(hpB[t], B[t], G, V[t], null, null, alpha, data.getiFeatureType());
				psU = initializePosteriorStatistics(hpU[t], U[t], F, A[t], R, V[t], alpha, data.getuFeatureType());
				psV = initializePosteriorStatistics(hpV[t], V[t], G, B[t], transpose(R), U[t], alpha, data.getiFeatureType());
			}
//			//Sample latent variables
			sampleLatentVariables(hpA[t], A, psA, t + 1, F, U[t], null, null, alpha, data.getuFeatureType());
			sampleLatentVariables(hpB[t], B, psB, t + 1, G, V[t], null, null, alpha, data.getiFeatureType());
			sampleLatentVariables(hpU[t], U, psU, t + 1, F, A[t + 1], R, V[t], alpha, data.getuFeatureType());
			sampleLatentVariables(hpV[t], V, psV, t + 1, G, B[t + 1], transpose(R), U[t + 1], alpha, data.getiFeatureType());

			//Sample latent variables
//			sampleLatentVariables(hpA[t], A, t + 1, F, U[t], null, null, alpha, data.getuFeatureType());
//			sampleLatentVariables(hpB[t], B, t + 1, G, V[t], null, null, alpha, data.getiFeatureType());
//			sampleLatentVariables(hpU[t], U, t + 1, F, A[t + 1], R, V[t], alpha, data.getuFeatureType());
//			sampleLatentVariables(hpV[t], V, t + 1, G, B[t + 1], transpose(R), U[t + 1], alpha, data.getiFeatureType());
		}
		
		Matrix result = new no.uib.cipr.matrix.DenseMatrix(m, n);
		for(int i = numBuffer + 1; i < U.length; i++) {
			result.add(matrixMult(U[i], transpose(V[i])));
		}
        return new MatrixFactorizationResult(result.scale(1.0/numSamples), U, V);
    }
    
    static void sampleHyperparams(Matrix Fin, Statistics[] hp, int t) {
    	DenseMatrix F = mfactory.copyArray(Matrices.getArray(Fin));
    	DenseVector Fbar = average(F); //Sample mean
    	DenseMatrix S = covariance(F, Fbar); //Sample covariance
    	int latentDim = F.getNumColumns();
    	int n = F.getNumRows();
    	
    	DenseMatrix Psi = (DenseMatrix) S.plus(mfactory.createIdentity(latentDim, latentDim)).plus(Fbar.outerProduct(Fbar).scale(n/(n + 1.0)));
    	InverseWishartDistribution iw = new InverseWishartDistribution(Psi, latentDim + n);
		DenseMatrix cov = (DenseMatrix) iw.sample(rand); //Sample covariance matrix from Wishart distribution
		MultivariateGaussian normal = new MultivariateGaussian(Fbar.scale(n / (1.0 + n)), cov.scale(1.0/(1 + n)));
		hp[t] = new Statistics(normal.sample(rand), cov); //Sample mean from Multivariate Gaussian distribution
    }
    
    static void sampleLatentVariables(Statistics hpin, Matrix[] F, int t, Matrix RF, Matrix Ain, Matrix R, Matrix Ein, double alpha, ArrayList<String> featureType) {
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
					if (R.get(i, j) > 0) {
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

	static void sampleLatentVariables(Statistics hpin, Matrix[] F,
									  Statistics[] ps, int t, Matrix RF, Matrix Ain, Matrix R, Matrix Ein,
									  double alpha, ArrayList<String> featureType) {
		F[t] = new no.uib.cipr.matrix.DenseMatrix(F[t-1].numRows(), F[t-1].numColumns());
		DenseMatrix oldF = mfactory.copyArray(Matrices.getArray(F[t-1]));
		DenseMatrix A = mfactory.copyArray(Matrices.getArray(Ain));
		DenseMatrix E = null;
		if(Ein != null) E = mfactory.copyArray(Matrices.getArray(Ein));

		for(int i = 0; i < F[t].numRows(); i++) {
			MultivariateGaussian oldDist = new MultivariateGaussian(ps[i].mu, ps[i].cov);
			DenseVector newFi = (DenseVector) oldDist.sample(rand);
			Statistics postStat = getPosteriorStatistics(hpin, newFi, i, RF, A, R, E, alpha, featureType); // Here mu = grad, cov = - Hessian inverse
			double eta = rand.nextDouble();
			postStat.mu = (DenseVector) newFi.minus(postStat.cov.times(postStat.mu).scale(eta));

			DenseVector oldFi = oldF.getRow(i);
			System.out.println("new: " + newFi);
			System.out.println("old: " + oldFi);
//			System.out.println(getLogPosteriorProbability(hpin, newFi, i, RF, A, R, E, alpha, featureType));
//			System.out.println(getLogPosteriorProbability(hpin, oldFi, i, RF, A, R, E, alpha, featureType));
			double acceptanceProbability = getLogPosteriorProbability(hpin, newFi, i, RF, A, R, E, alpha, featureType)
					- getLogPosteriorProbability(hpin, oldFi, i, RF, A, R, E, alpha, featureType);
			MultivariateGaussian newDist = new MultivariateGaussian(postStat.mu, postStat.cov);
			acceptanceProbability = Math.exp(acceptanceProbability) * newDist.getProbabilityFunction().evaluate(oldFi)
					/ oldDist.getProbabilityFunction().evaluate(newFi);

			if(rand.nextDouble() < acceptanceProbability) {
//				System.out.println(1);
				ps[i] = postStat;
				setRow(F[t], newFi, i);
			} else {
//				System.out.println(0);
				setRow(F[t], oldFi, i);
			}
		}
	}

	static Statistics[] initializePosteriorStatistics(Statistics hpin, Matrix Fin, Matrix RF, Matrix Ain, Matrix R,
													  Matrix Ein, double alpha, ArrayList<String> featureType) {
		Statistics[] result = new Statistics[Fin.numRows()];
		DenseMatrix F = mfactory.copyArray(Matrices.getArray(Fin));
		DenseMatrix A = mfactory.copyArray(Matrices.getArray(Ain));
		DenseMatrix E = null;
		if(Ein != null) E = mfactory.copyArray(Matrices.getArray(Ein));
		for(int i = 0; i < Fin.numRows(); i++) {
			result[i] = initializePosteriorStatistics(hpin, F.getRow(i), i, RF, A, R, E, alpha, featureType);
		}
		return result;
	}

	static Statistics initializePosteriorStatistics(Statistics hpin, DenseVector F, int id, Matrix RF, DenseMatrix A,
													Matrix R, DenseMatrix E, double alpha, ArrayList<String> featureType) {
		Statistics result = getPosteriorStatistics(hpin, F, id, RF, A, R, E, alpha, featureType);
		result.mu = (DenseVector) F.minus(result.cov.times(result.mu).scale(rand.nextDouble()));
		return result;
	}

	static Statistics getPosteriorStatistics(Statistics hpin, DenseVector F, int id, Matrix RF, DenseMatrix A,
											 Matrix R, DenseMatrix E, double alpha, ArrayList<String> featureType) {
		DenseVector pMu = vfactory.createVector(F.getDimensionality());
		DenseMatrix pPrec = mfactory.createMatrix(F.getDimensionality(), F.getDimensionality());
		for(int j = 0; j < A.getNumRows(); j++) {
			DenseVector Aj = A.getRow(j);
			int featureIndex = id;
			if(E != null) featureIndex = j;

			double Fij = 0;
			if (E == null) Fij = RF.get(j, id);
			else Fij = RF.get(id, j);

			if(featureType.get(featureIndex).equals("b")) {
				double expDot = Math.exp(-1 * F.dotProduct(Aj));
//				System.out.println(expDot);
				pMu.plusEquals(Aj.scale(Fij - 1 / (1 + expDot)));
				pPrec.plusEquals(Aj.outerProduct(Aj).scale(-1 * expDot / Math.pow(1 + expDot, 2)));
			} else {
				pMu.plusEquals(Aj.scale(alpha * (Fij - F.dotProduct(Aj))));
				pPrec.minusEquals(Aj.outerProduct(Aj).scale(alpha));
			}
		}

		gov.sandia.cognition.math.matrix.Matrix prec = hpin.cov.inverse();
		pMu.minusEquals(prec.times(F.minus(hpin.mu)));
		pPrec.minusEquals(prec);

		if(E != null) {
			for(int j = 0; j < R.numColumns(); j++) {
				if(R.get(id, j) > 0) {
					DenseVector Ej = E.getRow(j);
					pMu.plusEquals(Ej.scale(alpha * (R.get(id, j) - F.dotProduct(Ej))));
					pPrec.minusEquals(Ej.outerProduct(Ej).scale(alpha));
				}
			}
		}

		return new Statistics(pMu, pPrec.inverse().scale(-1));
	}

	static double getLogPosteriorProbability(Statistics hpin, DenseVector F, int id, Matrix RF, DenseMatrix A, Matrix R, DenseMatrix E, double alpha, ArrayList<String> featureType) {
		double result = 0;
		for(int j = 0; j < A.getNumRows(); j++) {
			int featureIndex = id;
			if(E != null) featureIndex = j;

			DenseVector Aj = A.getRow(j);
			double Fij = 0;
			if(E == null) Fij = RF.get(j, id);
			else Fij = RF.get(id, j);

			double dotProd = F.dotProduct(Aj);
			if(featureType.get(featureIndex).equals("b")) {
				result += Fij * dotProd - Math.log(1 + Math.exp(dotProd));
			} else {
				result -= 0.5 * alpha * Math.pow(Fij - dotProd, 2);
			}
		}

		DenseMatrix prec = (DenseMatrix) hpin.cov.inverse();
		DenseVector FMinusMu = (DenseVector) F.minus(hpin.mu);
		result -= 0.5 * FMinusMu.dotProduct(prec.times(FMinusMu));

		if(E != null) {
			for(int j = 0; j < R.numColumns(); j++) {
				if(R.get(id, j) > 0) {
					DenseVector Ej = E.getRow(j);
					result -= 0.5 * alpha * Math.pow(R.get(id, j) - F.dotProduct(Ej), 2);
				}
			}
		}

		return result;
	}

//	static void initializeHessianMH(Hyperparams hStatistics[] F, int t, Matrix RF, Matrix Ain, Matrix R, Matrix Ein, double alpha) {
//
//	}
    
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
                x.set(i, j, r.nextDouble() - 1);
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