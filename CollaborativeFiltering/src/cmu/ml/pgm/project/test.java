package cmu.ml.pgm.project;
/**
 * Created by dexter on 15. 3. 3..
 */
public class test {
    public static void main(String[] args) {
        MatrixFactorizationMovieLens mf
                = new MatrixFactorizationMovieLens("Data/ml-100k/u.user", "Data/ml-100k/u.item",
                "Data/ml-100k/u.data", "Data/ml-100k/u.info");

        mf.printMatrix();
    }
}
