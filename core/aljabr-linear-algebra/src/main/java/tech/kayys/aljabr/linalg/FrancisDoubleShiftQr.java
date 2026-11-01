package tech.kayys.aljabr.linalg;

/**
 * One implicit Francis double-shift QR bulge-chasing sweep on the leading
 * active block of an upper-Hessenberg matrix.
 *
 * <p>The shift polynomial is formed from the trailing 2x2 block. Orthogonal
 * Householder reflectors are applied from both sides and accumulated into Q,
 * so the similarity relation A = Q*T*Q^T is preserved.</p>
 */
final class FrancisDoubleShiftQr {
    private FrancisDoubleShiftQr() {}

    static void sweep(double[][] h, double[][] q, int active) {
        if (active < 3) throw new IllegalArgumentException("Francis sweep requires active order >= 3");

        double a = h[active - 2][active - 2];
        double b = h[active - 2][active - 1];
        double c = h[active - 1][active - 2];
        double d = h[active - 1][active - 1];
        double shiftSum = a + d;
        double shiftProduct = Math.fma(a, d, -b * c);

        double h00 = h[0][0], h01 = h[0][1], h10 = h[1][0], h11 = h[1][1];
        double x = Math.fma(h00, h00, h01 * h10) - shiftSum * h00 + shiftProduct;
        double y = h10 * (h00 + h11 - shiftSum);
        double z = h10 * h[2][1];
        double scale = Math.max(Math.abs(x), Math.max(Math.abs(y), Math.abs(z)));
        if (scale == 0.0) return;
        x /= scale;
        y /= scale;
        z /= scale;

        for (int k = 0; k < active - 1; k++) {
            int length = Math.min(3, active - k);
            double[] v = new double[length];
            if (k == 0) {
                v[0] = x;
                v[1] = y;
                v[2] = z;
            } else {
                for (int i = 0; i < length; i++) v[i] = h[k + i][k - 1];
            }

            double norm = 0.0;
            for (double value : v) norm = Math.hypot(norm, value);
            if (norm == 0.0) continue;
            double alpha = -Math.copySign(norm, v[0]);
            v[0] -= alpha;
            double vNorm = 0.0;
            for (double value : v) vNorm = Math.hypot(vNorm, value);
            if (vNorm == 0.0) continue;
            for (int i = 0; i < length; i++) v[i] /= vNorm;

            // H <- P*H, including columns to the right of the active block so
            // that the similarity remains valid for a deflated trailing block.
            for (int column = Math.max(0, k - 1); column < h.length; column++) {
                double projection = 0.0;
                for (int i = 0; i < length; i++)
                    projection = Math.fma(v[i], h[k + i][column], projection);
                for (int i = 0; i < length; i++)
                    h[k + i][column] = Math.fma(-2.0 * v[i], projection, h[k + i][column]);
            }

            // H <- H*P.
            for (int row = 0; row < h.length; row++) {
                double projection = 0.0;
                for (int i = 0; i < length; i++)
                    projection = Math.fma(h[row][k + i], v[i], projection);
                for (int i = 0; i < length; i++)
                    h[row][k + i] = Math.fma(-2.0 * projection, v[i], h[row][k + i]);
            }

            // Q <- Q*P.
            for (int row = 0; row < q.length; row++) {
                double projection = 0.0;
                for (int i = 0; i < length; i++)
                    projection = Math.fma(q[row][k + i], v[i], projection);
                for (int i = 0; i < length; i++)
                    q[row][k + i] = Math.fma(-2.0 * projection, v[i], q[row][k + i]);
            }

            if (k > 0) {
                for (int row = k + 1; row < Math.min(active, k + length); row++)
                    h[row][k - 1] = 0.0;
            }
        }
    }
}
