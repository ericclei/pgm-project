package cmu.ml.pgm.project;


import gov.sandia.cognition.math.matrix.mtj.DenseMatrix;
import gov.sandia.cognition.math.matrix.mtj.DenseMatrixFactoryMTJ;
import gov.sandia.cognition.math.matrix.mtj.DenseVector;
import gov.sandia.cognition.math.matrix.mtj.DenseVectorFactoryMTJ;
import gov.sandia.cognition.statistics.distribution.InverseWishartDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
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
//    public static MatrixFactorizationResult factorizeMatrixWithFeatures(MatrixFactorizationMovieLens data,
//                                                                        int latentDim, int numSamples, double alpha) {
//		Matrix R = data.getRelations(0, 1);
//		Matrix F = data.getuFeatureMatrix();
//		Matrix G = data.getiFeatureMatrix();
//		int dF = F.numColumns();
//		int dG = G.numColumns();
//		int m = F.numRows();
//		int n = G.numRows();
//		int numBuffer = 0;
//		int length = numSamples + numBuffer + 1;
//		Matrix[] U = new Matrix[length];
//		U[0] = new no.uib.cipr.matrix.DenseMatrix(m, latentDim);
//		Matrix[] V = new Matrix[length];
//		V[0] = new no.uib.cipr.matrix.DenseMatrix(n, latentDim);
//		Matrix[] A = new Matrix[length];
//		A[0] = new no.uib.cipr.matrix.DenseMatrix(dF, latentDim);
//		Matrix[] B = new Matrix[length];
//		B[0] = new no.uib.cipr.matrix.DenseMatrix(dG, latentDim);
//		randomlyInitialize(U[0]);
//		randomlyInitialize(V[0]);
//		randomlyInitialize(A[0]);
//		randomlyInitialize(B[0]);
//
//		Statistics[] hpU = new Statistics[length];
//		Statistics[] hpV = new Statistics[length];
//		Statistics[] hpA = new Statistics[length];
//		Statistics[] hpB = new Statistics[length];
//
//		hpU[0] = new Statistics(vfactory.createVector(latentDim), mfactory.createIdentity(latentDim, latentDim));
//		hpV[0] = new Statistics(vfactory.createVector(latentDim), mfactory.createIdentity(latentDim, latentDim));
//		hpA[0] = new Statistics(vfactory.createVector(latentDim), mfactory.createIdentity(latentDim, latentDim));
//		hpB[0] = new Statistics(vfactory.createVector(latentDim), mfactory.createIdentity(latentDim, latentDim));
//
//		Statistics[] psU = null, psV = null, psA = null, psB = null;
//
//		psA = initializePosteriorStatistics(hpA[0], A[0], F, U[0], null, null, alpha, data.getuFeatureType());
//		psB = initializePosteriorStatistics(hpB[0], B[0], G, V[0], null, null, alpha, data.getiFeatureType());
//		psU = initializePosteriorStatistics(hpU[0], U[0], F, A[0], R, V[0], alpha, data.getuFeatureType());
//		psV = initializePosteriorStatistics(hpV[0], V[0], G, B[0], transpose(R), U[0], alpha, data.getiFeatureType());
//
////		numSamples = 1;
//		for(int t = 0; t < numSamples + numBuffer; t++) {
//			System.out.println("Sample: " + (t + 1));
//			//Sample HyperparStatisticse done in parallel
////			sampleHyperparams(U[t], hpU, t);
////			sampleHyperparams(V[t], hpV, t);
////			sampleHyperparams(A[t], hpA, t);
////			sampleHyperparams(B[t], hpB, t);
//
////			if(t == 0) {
////				psA = initializePosteriorStatistics(hpA[t], A[t], F, U[t], null, null, alpha, data.getuFeatureType());
////				psB = initializePosteriorStatistics(hpB[t], B[t], G, V[t], null, null, alpha, data.getiFeatureType());
////				psU = initializePosteriorStatistics(hpU[t], U[t], F, A[t], R, V[t], alpha, data.getuFeatureType());
////				psV = initializePosteriorStatistics(hpV[t], V[t], G, B[t], transpose(R), U[t], alpha, data.getiFeatureType());
////			}
////			//Sample latent variables
//			sampleLatentVariables(hpA[t], A, psA, t + 1, F, U[t], null, null, alpha, data.getuFeatureType());
//			sampleLatentVariables(hpB[t], B, psB, t + 1, G, V[t], null, null, alpha, data.getiFeatureType());
//			sampleLatentVariables(hpU[t], U, psU, t + 1, F, A[t + 1], R, V[t], alpha, data.getuFeatureType());
//			sampleLatentVariables(hpV[t], V, psV, t + 1, G, B[t + 1], transpose(R), U[t + 1], alpha, data.getiFeatureType());
//
//			sampleHyperparams(U[t + 1], hpU, t + 1);
//			sampleHyperparams(V[t + 1], hpV, t + 1);
//			sampleHyperparams(A[t + 1], hpA, t + 1);
//			sampleHyperparams(B[t + 1], hpB, t + 1);
//		}
//
//		Matrix result = new no.uib.cipr.matrix.DenseMatrix(m, n);
//		for(int i = numBuffer + 1; i < U.length; i++) {
//			result.add(matrixMult(U[i], transpose(V[i])));
//		}
//        return new MatrixFactorizationResult(result.scale(1.0/numSamples), U, V);
//    }



	public static CollectiveMatrixFactorizationResult factorizeMatrixWithFeatures(CollectiveMatrixFactorizationDataset data,
																		int latentDim, int numSamples, double alpha) {
		int nEntities = data.getNumEntities();
//		Matrix[] U = new Matrix[numSamples];
//		Matrix[] V = new Matrix[numSamples];
		CollectiveMatrixFactorizationResult result = new CollectiveMatrixFactorizationResult(nEntities);
		int numBuffer = 0;
		int length = numSamples + numBuffer + 1;
		Matrix[][] Entities = new Matrix[nEntities][length];
		Matrix[][] Features = new Matrix[nEntities][length];

		for(int i = 0; i < nEntities; i++) {
			Entities[i][0] = new no.uib.cipr.matrix.DenseMatrix(data.getNumItems(i), latentDim);
			Features[i][0] = new no.uib.cipr.matrix.DenseMatrix(
					data.getNumNormalFeatures(i) + data.getNumBernoulliFeatures(i), latentDim);

			randomlyInitialize(Entities[i][0]);
			randomlyInitialize(Features[i][0]);
		}

		Statistics[][] hpEntities = new Statistics[nEntities][length];
		Statistics[][] hpFeatures = new Statistics[nEntities][length];

//		Statistics[] psU = null, psV = null, psA = null, psB = null;
		Statistics[][] psEntities = new Statistics[nEntities][];
		Statistics[][] psFeatures = new Statistics[nEntities][];

		int[] cur_iteration = new int[nEntities];

		for(int i = 0; i < nEntities; i++) {
			hpEntities[i][0] = new Statistics(vfactory.createVector(latentDim), mfactory.createIdentity(latentDim, latentDim));
			hpFeatures[i][0] = new Statistics(vfactory.createVector(latentDim), mfactory.createIdentity(latentDim, latentDim));

			psFeatures[i] = initializePosteriorStatistics(data, hpFeatures[i][0], i, Features, Entities, null, alpha);
			psEntities[i] = initializePosteriorStatistics(data, hpFeatures[i][0], i, Entities, Features, cur_iteration, alpha);
		}


//		numSamples = 1;


		for(int t = 0; t < numSamples + numBuffer; t++) {
			System.out.println("Sample: " + (t + 1));
			//Sample HyperparStatisticse done in parallel
			for(int i = 0; i < nEntities; i++) {
				sampleHyperparams(Entities[i][t], hpEntities[i], t);
				sampleHyperparams(Features[i][t], hpFeatures[i], t);
			}
			//Sample latent variables
			for(int i = 0; i < nEntities; i++) {
				sampleLatentVariables(data, hpFeatures[i][t], psFeatures[i], i, Features, t + 1, Entities, t, null, alpha);
			}

			for(int i = 0; i < nEntities; i++) {
				sampleLatentVariables(data, hpEntities[i][t], psEntities[i], i, Entities, t + 1, Features, t + 1, cur_iteration, alpha);
				cur_iteration[i]++;
			}

//			if(t > numBuffer) {
//				U[t - numBuffer] = Entities[0][t + 1];
//				V[t - numBuffer] = Entities[1][t + 1];
//			}
			for(int i = 0; i < nEntities; i++) {
				result.setLatentFeatures(i, Entities[i][t + 1]);
			}
//			result.addIntermediateRelations();
		}
		result.addIntermediateRelations();
		return result;//new MatrixFactorizationResult(matrixMult(U[U.length - 1], transpose(V[V.length - 1])), U, V);
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
    
    static void sampleLatentVariables(CollectiveMatrixFactorizationDataset data, Statistics hpin, Matrix[][] Ein,
									  int eid, int t_E, Matrix[][] Ain, int t_A, int[] cur_iteration, double alpha) {
    	Ein[eid][t_E] = new no.uib.cipr.matrix.DenseMatrix(Ein[eid][t_E-1].numRows(), Ein[eid][t_E-1].numColumns());
    	DenseMatrix A = mfactory.copyArray(Matrices.getArray(Ain[eid][t_A]));
    	DenseMatrix[] E = null;
		Matrix[] R = null;
		if(cur_iteration != null) {
			E = new DenseMatrix[data.getNumEntities()];
			R = new Matrix[data.getNumEntities()];
			for (int j = 0; j < data.getNumEntities(); j++) {
				if (j != eid && data.getRelations(eid, j) != null) {
					E[j] = mfactory.copyArray(Matrices.getArray(Ein[j][cur_iteration[j]]));
					R[j] = data.getRelations(eid, j);
				}
			}
		}
    	int latentDim = Ein[eid][t_E].numColumns();
    	DenseVector precTimesMu = (DenseVector) hpin.cov.inverse().times(hpin.mu);

    	DenseMatrix prec = mfactory.createMatrix(latentDim, latentDim);
    	for(int i = 0; i < A.getNumRows(); i++) {
    		prec.plusEquals(A.getRow(i).outerProduct(A.getRow(i)));
    	}
    	prec.scaleEquals(alpha);
    	prec.plusEquals(hpin.cov.inverse());

    	for(int i = 0; i < Ein[eid][t_E].numRows(); i++) {
    		DenseMatrix prec_ind = prec.clone();
    		DenseVector mu = (DenseVector) precTimesMu.clone();
    		for(int j = 0; j < A.getNumRows(); j++) {
    			double rf;
    			if(E == null) {
    				rf = data.getNormalFeatures(eid).get(j, i);
    			} else {
    				rf = data.getNormalFeatures(eid).get(i, j);
    			}
    			mu.plusEquals(A.getRow(j).scale(alpha * rf));
    		}
    		if(E != null) {
				for(int j = 0; j < E.length; j++) {
					if(E[j] != null) {
						for (int k = 0; k < E[j].getNumRows(); k++) {
							if (R[j].get(i, k) > 0) {
								DenseVector Ek = E[j].getRow(k);
								prec_ind.plusEquals(Ek.outerProduct(Ek).scale(alpha));
								mu.plusEquals(Ek.scale(alpha * R[j].get(i, k)));
							}
						}
					}
				}
    		}
    		DenseMatrix cov_ind = (DenseMatrix) prec_ind.inverse();
    		mu = (DenseVector) mu.times(cov_ind);
    		MultivariateGaussian normal = new MultivariateGaussian(mu, cov_ind);
    		setRow(Ein[eid][t_E], (DenseVector) normal.sample(rand), i);
    	}
    }

	static void sampleLatentVariables(CollectiveMatrixFactorizationDataset data, Statistics hpin, Statistics[] ps, int eid, Matrix[][] Ein,
									  int t_E, Matrix[][] Ain, int t_A, int[] cur_iteration, double alpha) {
		Ein[eid][t_E] = new no.uib.cipr.matrix.DenseMatrix(Ein[eid][t_E-1].numRows(), Ein[eid][t_E-1].numColumns());
		DenseMatrix oldE = mfactory.copyArray(Matrices.getArray(Ein[eid][t_E-1]));
		DenseMatrix A = mfactory.copyArray(Matrices.getArray(Ain[eid][t_A]));
		DenseMatrix[] E = null;
		Matrix[] R = null;
		if(cur_iteration != null) {
			E = new DenseMatrix[data.getNumEntities()];
			R = new Matrix[data.getNumEntities()];
			for (int j = 0; j < data.getNumEntities(); j++) {
				if (j != eid && data.getRelations(eid, j) != null) {
					E[j] = mfactory.copyArray(Matrices.getArray(Ein[j][cur_iteration[j]]));
					R[j] = data.getRelations(eid, j);
				}
			}
		}

		for(int i = 0; i < Ein[eid][t_E].numRows(); i++) {
			MultivariateGaussian oldDist = new MultivariateGaussian(ps[i].mu, ps[i].cov);
			DenseVector newEi = (DenseVector) oldDist.sample(rand);
			Statistics postStat = getPosteriorStatistics(data, hpin, newEi, eid, i, A, R, E, alpha); // Here mu = grad, cov = - Hessian inverse
			double eta = rand.nextDouble();
			postStat.mu = (DenseVector) newEi.plus(postStat.cov.times(postStat.mu).scale(eta));

			DenseVector oldEi = oldE.getRow(i);
			double acceptanceProbability = getLogPosteriorProbability(data, hpin, newEi, eid, i, A, R, E, alpha)
					- getLogPosteriorProbability(data, hpin, oldEi, eid, i, A, R, E, alpha);
			MultivariateGaussian newDist = new MultivariateGaussian(postStat.mu, postStat.cov);
			acceptanceProbability = Math.exp(acceptanceProbability) * newDist.getProbabilityFunction().evaluate(oldEi)
					/ oldDist.getProbabilityFunction().evaluate(newEi);

			if(rand.nextDouble() < acceptanceProbability) {
//				System.out.println(1);
				ps[i] = postStat;
				setRow(Ein[eid][t_E], newEi, i);
			} else {
//				System.out.println(0);
				setRow(Ein[eid][t_E], oldEi, i);
			}
		}
	}

	static Statistics[] initializePosteriorStatistics(CollectiveMatrixFactorizationDataset data, Statistics hpin, int eid,
													  Matrix[][] Ein, Matrix[][] Ain, int[] cur_iteration, double alpha) {
		Statistics[] result = new Statistics[Ein[eid][0].numRows()];
		DenseMatrix F = mfactory.copyArray(Matrices.getArray(Ein[eid][0]));
		DenseMatrix A = mfactory.copyArray(Matrices.getArray(Ain[eid][0]));
		DenseMatrix[] E = null;
		Matrix[] R = null;
		if(cur_iteration != null) {
			E = new DenseMatrix[data.getNumEntities()];
			R = new Matrix[data.getNumEntities()];
			for (int j = 0; j < data.getNumEntities(); j++) {
				if (j != eid && data.getRelations(eid, j) != null) {
					E[j] = mfactory.copyArray(Matrices.getArray(Ein[j][cur_iteration[j]]));
					R[j] = data.getRelations(eid, j);
				}
			}
		}

		for(int i = 0; i < F.getNumRows(); i++) {
			result[i] = initializePosteriorStatistics(data, hpin, F.getRow(i), eid, i, A, R, E, alpha);
		}
		return result;
	}

	static Statistics initializePosteriorStatistics(CollectiveMatrixFactorizationDataset data, Statistics hpin, DenseVector F, int eid,
													int id, DenseMatrix A, Matrix[] R, DenseMatrix[] E, double alpha) {
		Statistics result = getPosteriorStatistics(data, hpin, F, eid, id, A, R, E, alpha);
		double eta = rand.nextDouble();
		result.mu = (DenseVector) F.minus(result.cov.times(result.mu).scale(eta));
		return result;
	}

	static Statistics getPosteriorStatistics(CollectiveMatrixFactorizationDataset data, Statistics hpin, DenseVector F, int eid,
											 int id, DenseMatrix A, Matrix[] R, DenseMatrix[] E, double alpha) {
		int latentDim = F.getDimensionality();
		DenseVector pMu = vfactory.createVector(latentDim);
		DenseMatrix pPrec = mfactory.createMatrix(latentDim, latentDim);
		for(int j = 0; j < A.getNumRows(); j++) {
			DenseVector Aj = A.getRow(j);
			int featureIndex = id;
			if(E != null) featureIndex = j;

			double Fij = 0;
			if (E == null) Fij = data.getFeatures(eid).get(j, id);
			else Fij = data.getFeatures(eid).get(id, j);

			if(data.getFeatureTypes(eid).get(featureIndex).equals("b")) {
				double expDot = Math.exp(-1 * F.dotProduct(Aj));
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
			for(int j = 0; j < R.length; j++) {
				if(E[j] != null) {
					for(int k = 0; k < R[j].numColumns(); k++) {
						if(R[j].get(id, k) > 0) {
							DenseVector Ek = E[j].getRow(k);
							pMu.plusEquals(Ek.scale(alpha * (R[j].get(id, k) - F.dotProduct(Ek))));
							pPrec.minusEquals(Ek.outerProduct(Ek).scale(alpha));
						}
					}
				}
			}
		}

		return new Statistics(pMu, pPrec.inverse().scale(-1));
	}

	static double getLogPosteriorProbability(CollectiveMatrixFactorizationDataset data, Statistics hpin, DenseVector F, int eid,
											 int id, DenseMatrix A, Matrix[] R, DenseMatrix[] E, double alpha) {
		double result = 0;
		for(int j = 0; j < A.getNumRows(); j++) {
			DenseVector Aj = A.getRow(j);
			int featureIndex = id;
			if(E != null) featureIndex = j;

			double Fij = 0;
			if (E == null) Fij = data.getFeatures(eid).get(j, id);
			else Fij = data.getFeatures(eid).get(id, j);

			double dotProd = F.dotProduct(Aj);
			if(data.getFeatureTypes(eid).get(featureIndex).equals("b")) {
				result += Fij * dotProd - Math.log(1 + Math.exp(dotProd));
			} else {
				result -= 0.5 * alpha * Math.pow(Fij - dotProd, 2);
			}
		}

		DenseMatrix prec = (DenseMatrix) hpin.cov.inverse();
		DenseVector FMinusMu = (DenseVector) F.minus(hpin.mu);
		result -= 0.5 * FMinusMu.dotProduct(prec.times(FMinusMu));

		if(E != null) {
			for(int j = 0; j < R.length; j++) {
				if(E[j] != null) {
					for(int k = 0; k < R[j].numColumns(); k++) {
						if(R[j].get(id, k) > 0) {
							DenseVector Ek = E[j].getRow(k);
							result -= 0.5 * alpha * Math.pow(R[j].get(id, k) - F.dotProduct(Ek), 2);
						}
					}
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
    
    public static Matrix matrixMult(Matrix x, Matrix y) {
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
                x.set(i, j, r.nextGaussian());
    }

    static void setAllValues(Matrix x, double val) {
        for (int i = 0; i < x.numRows(); i++)
            for (int j = 0; j < x.numColumns(); j++)
                x.set(i, j, val);
    }

    public static Matrix transpose(Matrix x) {
        return x.transpose(new no.uib.cipr.matrix.DenseMatrix(x.numColumns(), x.numRows()));
    }

	public static void writeMatrices(
			Matrix[][] result, int t, String prefix) {
		try {
			String dir = "output/Bayesian";
			for (int s = 0; s < result.length; s++) {
				String filename = dir + String.format(prefix + ".%d.(%d).dat", s, t);
				File file = new File(filename);
				FileWriter fw = new FileWriter(file.getAbsoluteFile());
				BufferedWriter bw = new BufferedWriter(fw);
				Matrix u = result[s][t];
				int nRows = u.numRows();
				int nCols = u.numColumns();
				for (int i = 0; i < nRows; i++) {
					bw.write("" + u.get(i, 0));
					for (int j = 1; j < nCols; j++) {
						bw.write("," + u.get(i, j));
					}
					bw.newLine();
				}
				bw.close();
			}
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}
}