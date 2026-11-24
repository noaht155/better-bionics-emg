using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.XR.Hands;

// The free hand from the headset's hand tracking (Multimodal.cs keeps it on while the other hand has a controller):
// wrist pose for the sim hand, the 20 joint angles from HandAngles, how far the fingers close, a pinch, and the
// index fingertip that works the menu by touch.
public class TrackedHand
{
    // Landmarks in MediaPipe's order from XR Hands' joints
    static readonly XRHandJointID[] Landmarks =
    {
        XRHandJointID.Wrist,
        XRHandJointID.ThumbMetacarpal, XRHandJointID.ThumbProximal, XRHandJointID.ThumbDistal, XRHandJointID.ThumbTip,
        XRHandJointID.IndexProximal, XRHandJointID.IndexIntermediate, XRHandJointID.IndexDistal, XRHandJointID.IndexTip,
        XRHandJointID.MiddleProximal, XRHandJointID.MiddleIntermediate, XRHandJointID.MiddleDistal, XRHandJointID.MiddleTip,
        XRHandJointID.RingProximal, XRHandJointID.RingIntermediate, XRHandJointID.RingDistal, XRHandJointID.RingTip,
        XRHandJointID.LittleProximal, XRHandJointID.LittleIntermediate, XRHandJointID.LittleDistal, XRHandJointID.LittleTip,
    };
    // Unity's joint frame points forward to the fingertips with up out of the back of the hand; the sim hand points
    // its fingers along local -z with the palm up, so it's turned half round x
    static readonly Quaternion ToSimHand = Quaternion.Euler(180, 0, 0);
    const float PinchOnM = 0.015f, PinchOffM = 0.03f;

    readonly List<XRHandSubsystem> subsystems = new List<XRHandSubsystem>();
    readonly Vector3[] points = new Vector3[21];
    public readonly float[] Angles = new float[20];
    public bool Tracked { get; private set; }
    public Vector3 WristPos { get; private set; }
    public Quaternion WristRot { get; private set; }
    public Vector3 IndexTip { get; private set; }
    public Vector3 PalmNormal { get; private set; }
    public Vector3 PinchPoint { get; private set; }
    // The index knuckle: stays put while pinching, so the pointing ray goes through it
    public Vector3 IndexKnuckle { get; private set; }
    // 0 open to 1 a fist, from the middle, ring and little fingers
    public float Closure { get; private set; }
    public bool Pinching { get; private set; }
    // Thumb on the middle fingertip, the menu gesture with the palm turned to the face
    public bool MiddlePinching { get; private set; }

    // The sim's hand on each side for fitting the thumb (export_model.ghost_xml)
    readonly IntPtr[] fk = new IntPtr[2];
    readonly float[] simPoints = new float[63];
    readonly double[] trial = new double[20];
    readonly float[] thumb = { 0, 20, 10, 10 };
    static readonly float[] ThumbMin = { -30, -15, -10, -10 }, ThumbMax = { 80, 80, 80, 90 };
    // Sim wrist to middle knuckle (m), what a tracked hand is scaled to
    float simPalm;

    public TrackedHand(string ghostLeft, string ghostRight)
    {
        var err = new byte[500];
        for (int i = 0; i < 2; i++)
        {
            var xml = Encoding.UTF8.GetBytes(i == 1 ? ghostRight : ghostLeft);
            fk[i] = Native.bb_fk_create(xml, xml.Length, err, err.Length);
        }
        if (fk[1] == IntPtr.Zero) return;
        Native.bb_fk_points(fk[1], trial, simPoints);
        simPalm = new Vector3(simPoints[27], simPoints[28], simPoints[29]).magnitude;
    }

    public void Update(bool rightHand)
    {
        Tracked = false;
        if (subsystems.Count == 0) SubsystemManager.GetSubsystems(subsystems);
        if (subsystems.Count == 0 || !subsystems[0].running) return;
        var hand = rightHand ? subsystems[0].rightHand : subsystems[0].leftHand;
        if (!hand.isTracked) return;
        for (int i = 0; i < Landmarks.Length; i++)
        {
            if (!hand.GetJoint(Landmarks[i]).TryGetPose(out var pose)) return;
            // Unity's z forward is OpenXR's -z: flipped back into a right-handed frame for the angles
            points[i] = new Vector3(pose.position.x, pose.position.y, -pose.position.z);
            if (i == 0)
            {
                WristPos = pose.position;
                WristRot = pose.rotation * ToSimHand;
                PalmNormal = pose.rotation * Vector3.down;
            }
        }
        HandAngles.Compute(points, rightHand, Angles);
        StableSpread();
        FitThumb(rightHand);
        float bend = 0;
        for (int f = 2; f < 5; f++) bend += Mathf.Clamp01((Angles[4 * f] + Angles[4 * f + 2]) / 160f);
        Closure = bend / 3;
        var thumbTip = Unflip(points[4]);
        IndexTip = Unflip(points[8]);
        PinchPoint = (thumbTip + IndexTip) / 2;
        IndexKnuckle = Unflip(points[5]);
        float gap = Vector3.Distance(thumbTip, IndexTip);
        Pinching = Pinching ? gap < PinchOffM : gap < PinchOnM;
        gap = Vector3.Distance(thumbTip, Unflip(points[12]));
        MiddlePinching = MiddlePinching ? gap < PinchOffM : gap < PinchOnM;
        Tracked = true;
    }

    static Vector3 Unflip(Vector3 p) => new Vector3(p.x, p.y, -p.z);

    // A finger curled towards the palm has almost no sideways direction left, so its spread angle is noise that
    // threw the ring and little fingers about in a fist: faded out as the base knuckle bends
    void StableSpread()
    {
        for (int f = 1; f < 5; f++)
            Angles[4 * f + 1] *= Mathf.Clamp01(Mathf.Cos(Angles[4 * f] * Mathf.Deg2Rad));
    }

    // The sim thumb's joints aren't the real thumb's (no turn of the metacarpal), so its angles from hand_angles put
    // the tip off to the side. Instead its four angles are fitted so the sim thumb's tip and last joint land where the
    // real ones are in the palm's frame, scaled to the sim hand: a few damped Gauss-Newton steps from last frame's fit
    void FitThumb(bool rightHand)
    {
        var h = fk[rightHand ? 1 : 0];
        if (h == IntPtr.Zero || simPalm <= 0) return;
        var y = Unit(points[9] - points[0]);
        var x = points[5] - points[17];
        x = Unit(x - Vector3.Dot(x, y) * y);
        var z = Cross(x, y);
        if (!rightHand) z = -z;
        float scale = simPalm / (points[9] - points[0]).magnitude, m = rightHand ? 1 : -1;
        Vector3 InSim(int i)
        {
            var v = (points[i] - points[0]) * scale;
            return new Vector3(m * Vector3.Dot(v, x), Vector3.Dot(v, y), Vector3.Dot(v, z));
        }
        Vector3 tipGoal = InSim(4), ipGoal = InSim(3);
        for (int j = 0; j < 20; j++) trial[j] = Angles[j];
        var r = new float[6];
        var jac = new float[6, 4];
        for (int iter = 0; iter < 3; iter++)
        {
            Residual(h, thumb, tipGoal, ipGoal, r);
            for (int k = 0; k < 4; k++)
            {
                var moved = (float[])thumb.Clone();
                moved[k] += 1;
                var rk = new float[6];
                Residual(h, moved, tipGoal, ipGoal, rk);
                for (int i = 0; i < 6; i++) jac[i, k] = rk[i] - r[i];
            }
            // (JtJ + damping) step = -Jt r, in degrees
            var a = new float[4, 5];
            for (int p = 0; p < 4; p++)
            {
                for (int q = 0; q < 4; q++)
                {
                    float sum = 0;
                    for (int i = 0; i < 6; i++) sum += jac[i, p] * jac[i, q];
                    a[p, q] = sum + (p == q ? 1e-6f : 0);
                }
                float rhs = 0;
                for (int i = 0; i < 6; i++) rhs -= jac[i, p] * r[i];
                a[p, 4] = rhs;
            }
            var step = Solve4(a);
            for (int k = 0; k < 4; k++)
                thumb[k] = Mathf.Clamp(thumb[k] + Mathf.Clamp(step[k], -20, 20), ThumbMin[k], ThumbMax[k]);
        }
        for (int k = 0; k < 4; k++) Angles[k] = thumb[k];
    }

    void Residual(IntPtr h, float[] t, Vector3 tipGoal, Vector3 ipGoal, float[] r)
    {
        for (int k = 0; k < 4; k++) trial[k] = t[k];
        Native.bb_fk_points(h, trial, simPoints);
        var tip = new Vector3(simPoints[12], simPoints[13], simPoints[14]) - tipGoal;
        var ip = (new Vector3(simPoints[9], simPoints[10], simPoints[11]) - ipGoal) * 0.5f;
        r[0] = tip.x; r[1] = tip.y; r[2] = tip.z; r[3] = ip.x; r[4] = ip.y; r[5] = ip.z;
    }

    static float[] Solve4(float[,] a)
    {
        for (int c = 0; c < 4; c++)
        {
            int best = c;
            for (int i = c + 1; i < 4; i++) if (Mathf.Abs(a[i, c]) > Mathf.Abs(a[best, c])) best = i;
            for (int k = 0; k < 5; k++) (a[c, k], a[best, k]) = (a[best, k], a[c, k]);
            if (Mathf.Abs(a[c, c]) < 1e-12f) return new float[4];
            for (int i = 0; i < 4; i++)
            {
                if (i == c) continue;
                float f = a[i, c] / a[c, c];
                for (int k = c; k < 5; k++) a[i, k] -= f * a[c, k];
            }
        }
        return new[] { a[0, 4] / a[0, 0], a[1, 4] / a[1, 1], a[2, 4] / a[2, 2], a[3, 4] / a[3, 3] };
    }

    static Vector3 Unit(Vector3 v) => v.magnitude > 1e-9f ? v / v.magnitude : v;

    static Vector3 Cross(Vector3 a, Vector3 b) =>
        new Vector3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
