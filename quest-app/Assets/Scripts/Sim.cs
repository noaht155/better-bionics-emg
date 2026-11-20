using System;
using System.Diagnostics;
using System.Text;
using System.Threading;

// The contact sim on its own thread, stepped in real time like tools.Contact on the laptop. The main thread only
// stages inputs and reads the newest poses, both under the lock; every bbsim call happens on this thread.
public class Sim
{
    public const int Joints = 20;
    public const int ReadoutSize = 13;
    // Faster than the laptop's 60 Hz so the hand moves smoothly at the headset's frame rate
    const double TickS = 1.0 / 90;
    const double MaxStepS = 0.05;

    public class Geom
    {
        public int type, role, finger;
        public double[] size = new double[3];
        public float[] vert;
        public int[] face;
    }

    readonly object gate = new object();
    Thread thread;
    volatile bool running;
    IntPtr world;

    readonly double[] angles = new double[Joints];
    double effort;
    bool locked;
    readonly double[] handPos = new double[3], handQuat = new double[4];
    bool handStaged, resetStaged, paramsStaged;
    double massScale = 1, maxForce = 15;
    bool assist;
    byte[] xmlStaged;
    string objectStaged;

    float[] pos = new float[0], mat = new float[0];
    readonly double[] readout = new double[ReadoutSize];
    readonly double[] hand = new double[7];

    // Set by the sim thread, read under the lock
    public Geom[] Geoms { get; private set; }
    public int Version { get; private set; }
    public string Object { get; private set; }
    public string Error { get; private set; }
    // Sim time per second spent stepping, the headroom left
    public double Speed { get; private set; }

    public void Start()
    {
        running = true;
        thread = new Thread(Run) { IsBackground = true, Name = "sim" };
        thread.Start();
    }

    public void Stop()
    {
        running = false;
        thread?.Join(500);
    }

    public void Load(string obj, string xml)
    {
        lock (gate) { xmlStaged = Encoding.UTF8.GetBytes(xml); objectStaged = obj; paramsStaged = true; }
    }

    public void SetInputs(float[] a, float e, bool l)
    {
        lock (gate)
        {
            for (int j = 0; j < Joints; j++) angles[j] = a[j];
            effort = e;
            locked = l;
        }
    }

    public void SetParams(double mass, double force, bool assistOn)
    {
        lock (gate) { massScale = mass; maxForce = force; assist = assistOn; paramsStaged = true; }
    }

    public void SetHand(double[] p, double[] q)
    {
        lock (gate) { Array.Copy(p, handPos, 3); Array.Copy(q, handQuat, 4); handStaged = true; }
    }

    public void ResetObject()
    {
        lock (gate) resetStaged = true;
    }

    // Copies the newest geom poses, readout and hand pose (position then w x y z); false before the first world
    public bool Read(ref float[] p, ref float[] m, double[] r, double[] h)
    {
        lock (gate)
        {
            if (Geoms == null) return false;
            if (p.Length != pos.Length) { p = new float[pos.Length]; m = new float[mat.Length]; }
            Array.Copy(pos, p, pos.Length);
            Array.Copy(mat, m, mat.Length);
            Array.Copy(readout, r, ReadoutSize);
            Array.Copy(hand, h, 7);
            return true;
        }
    }

    void Build(byte[] xml, string obj)
    {
        var err = new byte[1000];
        IntPtr next = Native.bb_create(xml, xml.Length, err, err.Length);
        if (next == IntPtr.Zero)
        {
            Error = Encoding.UTF8.GetString(err).TrimEnd('\0');
            return;
        }
        if (world != IntPtr.Zero)
        {
            // The new object's world takes over the hand where it was, as sim.World.set_params does
            var p = new double[3];
            var q = new double[4];
            Native.bb_hand(world, p, q);
            Native.bb_set_hand(next, p, q, 1);
            Native.bb_free(world);
        }
        world = next;
        int n = Native.bb_ngeom(world);
        var geoms = new Geom[n];
        for (int g = 0; g < n; g++)
        {
            var geom = geoms[g] = new Geom();
            Native.bb_geom(world, g, out geom.type, out geom.role, out geom.finger, geom.size);
            int nv = Native.bb_mesh(world, g, null, null, out int nf);
            if (nv > 0)
            {
                geom.vert = new float[3 * nv];
                geom.face = new int[3 * nf];
                Native.bb_mesh(world, g, geom.vert, geom.face, out nf);
            }
        }
        Geoms = geoms;
        Object = obj;
        Error = null;
        pos = new float[3 * n];
        mat = new float[9 * n];
        Version++;
    }

    void Run()
    {
        var clock = Stopwatch.StartNew();
        double last = clock.Elapsed.TotalSeconds, busy = 0, simmed = 0;
        var a = new double[Joints];
        var r = new double[ReadoutSize];
        var hp = new double[3];
        var hq = new double[4];
        float[] p = new float[0], m = new float[0];
        while (running)
        {
            double now = clock.Elapsed.TotalSeconds;
            double dt = Math.Min(now - last, MaxStepS);
            last = now;
            double e;
            bool l;
            lock (gate)
            {
                if (xmlStaged != null)
                {
                    Build(xmlStaged, objectStaged);
                    xmlStaged = null;
                    if (world != IntPtr.Zero) { p = new float[pos.Length]; m = new float[mat.Length]; }
                }
                if (world != IntPtr.Zero)
                {
                    if (paramsStaged) Native.bb_set_params(world, massScale, maxForce, assist ? 1 : 0);
                    if (resetStaged) Native.bb_reset_object(world);
                    if (handStaged) Native.bb_set_hand(world, handPos, handQuat, 0);
                    paramsStaged = resetStaged = handStaged = false;
                }
                Array.Copy(angles, a, Joints);
                e = effort;
                l = locked;
            }
            if (world == IntPtr.Zero)
            {
                Thread.Sleep(20);
                continue;
            }
            double t0 = clock.Elapsed.TotalSeconds;
            Native.bb_step(world, dt, a, e, l ? 1 : 0);
            Native.bb_poses(world, p, m);
            Native.bb_readout(world, r);
            Native.bb_hand(world, hp, hq);
            busy += clock.Elapsed.TotalSeconds - t0;
            simmed += dt;
            lock (gate)
            {
                Array.Copy(p, pos, p.Length);
                Array.Copy(m, mat, m.Length);
                Array.Copy(r, readout, ReadoutSize);
                Array.Copy(hp, 0, hand, 0, 3);
                Array.Copy(hq, 0, hand, 3, 4);
                if (simmed >= 1)
                {
                    Speed = simmed / Math.Max(busy, 1e-6);
                    simmed = busy = 0;
                }
            }
            double left = TickS - (clock.Elapsed.TotalSeconds - now);
            if (left > 0) Thread.Sleep((int)(left * 1000));
        }
        if (world != IntPtr.Zero) Native.bb_free(world);
        world = IntPtr.Zero;
    }
}
