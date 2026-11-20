using UnityEngine;

// emg-reading/hand_angles.py in C#: the 20 joint angles (degrees, JOINTS order) from 21 landmarks in MediaPipe's
// order and true handedness, so a tracked hand drives the sim hand with the same definitions as the camera's
// labels. Keep the two in step.
public static class HandAngles
{
    const float ThumbRestDeg = 45f;
    static readonly int[][] Chains = { new[] { 1, 2, 3, 4 }, new[] { 5, 6, 7, 8 }, new[] { 9, 10, 11, 12 }, new[] { 13, 14, 15, 16 }, new[] { 17, 18, 19, 20 } };

    static Vector3 Unit(Vector3 v) => v.magnitude > 1e-9f ? v / v.magnitude : v;

    // Right-handed vectors throughout, as in the Python
    static Vector3 Cross(Vector3 a, Vector3 b) => new Vector3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);

    static float Elevation(Vector3 v, Vector3 z) =>
        Mathf.Atan2(Vector3.Dot(v, z), (v - Vector3.Dot(v, z) * z).magnitude) * Mathf.Rad2Deg;

    static float Bend(Vector3 a, Vector3 b, Vector3 axis)
    {
        var c = Cross(a, b);
        float ang = Mathf.Atan2(c.magnitude, Vector3.Dot(a, b)) * Mathf.Rad2Deg;
        return Vector3.Dot(c, axis) >= 0 ? ang : -ang;
    }

    // p: 21 landmarks in a right-handed frame (metres). out20: JOINTS order
    public static void Compute(Vector3[] p, bool rightHand, float[] out20)
    {
        var y = Unit(p[9] - p[0]);
        var x = p[5] - p[17];
        x = Unit(x - Vector3.Dot(x, y) * y);
        var z = Cross(x, y);
        // A left hand is a mirror image, the same cross product points out of the back of the hand
        if (!rightHand) z = -z;
        for (int f = 1; f < 5; f++)
        {
            var c = Chains[f];
            Vector3 mcp = p[c[0]], pip = p[c[1]], dip = p[c[2]], tip = p[c[3]];
            Vector3 meta = mcp - p[0], prox = pip - mcp, mid = dip - pip, dist = tip - dip;
            var ahead = Unit(meta - Vector3.Dot(meta, z) * z);
            var thumbwards = Unit(x - Vector3.Dot(x, ahead) * ahead);
            var flat = prox - Vector3.Dot(prox, z) * z;
            int o = 4 * f;
            out20[o] = Elevation(prox, z) - Elevation(meta, z);
            out20[o + 1] = Mathf.Atan2(Vector3.Dot(flat, thumbwards), Vector3.Dot(flat, ahead)) * Mathf.Rad2Deg;
            var axis = Cross(ahead, z);
            out20[o + 2] = Bend(prox, mid, axis);
            out20[o + 3] = Bend(mid, dist, axis);
        }
        var t = Chains[0];
        Vector3 tm = p[t[1]] - p[t[0]], tp = p[t[2]] - p[t[1]], td = p[t[3]] - p[t[2]];
        out20[0] = ThumbRestDeg - Mathf.Atan2(Vector3.Dot(tm, x), Vector3.Dot(tm, y)) * Mathf.Rad2Deg;
        out20[1] = Elevation(tm, z);
        out20[2] = Bend(tm, tp, Cross(tm, tp));
        out20[3] = Bend(tp, td, Cross(tp, td));
    }
}
