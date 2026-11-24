using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text;
using System.Threading;

// The contact sim on its own thread, stepped in real time like tools.Contact on the laptop. The main thread stages
// each hand's target and inputs under the lock and queues one-off commands (place an object, settings); every bbsim
// call happens on this thread.
public class Sim
{
    public const int Joints = 20;
    public const int Prosthetic = 0, Free = 1;
    // Faster than the laptop's 60 Hz so the hands move smoothly at the headset's frame rate
    const double TickS = 1.0 / 90;
    const double MaxStepS = 0.05;

    public class Geom
    {
        public int type, role, hand, finger, obj, surface;
        public float friction;
        public double[] size = new double[3];
        public float[] vert;
        public int[] face;
    }

    class HandInput
    {
        public readonly double[] angles = new double[Joints], pos = new double[3], quat = { 1, 0, 0, 0 };
        public double effort;
        public bool locked, grab, placed, teleport;
    }

    readonly object gate = new object();
    readonly ConcurrentQueue<Action<IntPtr>> commands = new ConcurrentQueue<Action<IntPtr>>();
    readonly HandInput[] inputs = { new HandInput(), new HandInput() };
    Thread thread;
    volatile bool running;
    IntPtr world;
    byte[] xmlStaged;
    Action<IntPtr> afterBuild;

    float[] pos = new float[0], mat = new float[0];
    readonly double[] readout = new double[Native.ReadoutSize];
    double[] objects = new double[0];

    // Set by the sim thread, read under the lock
    public Geom[] Geoms { get; private set; }
    public int Version { get; private set; }
    public int ObjectCount { get; private set; }
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

    // A new scene (the prosthetic side changed); objects start parked, after sets it up once it's built
    public void Load(string xml, Action<IntPtr> after)
    {
        lock (gate)
        {
            xmlStaged = Encoding.UTF8.GetBytes(xml);
            afterBuild = after;
        }
    }

    public void Do(Action<IntPtr> command) => commands.Enqueue(command);

    public void SetInput(int h, float[] angles, float effort, bool locked, bool grab)
    {
        lock (gate)
        {
            var i = inputs[h];
            for (int j = 0; j < Joints; j++) i.angles[j] = angles[j];
            i.effort = effort;
            i.locked = locked;
            i.grab = grab;
        }
    }

    // Hand h's target in sim coordinates (quaternion w x y z); teleport jumps the palm there
    public void SetHand(int h, double[] p, double[] q, bool teleport = false)
    {
        lock (gate)
        {
            var i = inputs[h];
            Array.Copy(p, i.pos, 3);
            Array.Copy(q, i.quat, 4);
            i.placed = true;
            i.teleport |= teleport;
        }
    }

    // Copies the newest geom poses, readout and object poses (7 per object: position, w x y z); false before the
    // first world
    public bool Read(ref float[] p, ref float[] m, double[] r, ref double[] o)
    {
        lock (gate)
        {
            if (Geoms == null) return false;
            if (p.Length != pos.Length) { p = new float[pos.Length]; m = new float[mat.Length]; }
            if (o.Length != objects.Length) o = new double[objects.Length];
            Array.Copy(pos, p, pos.Length);
            Array.Copy(mat, m, mat.Length);
            Array.Copy(readout, r, readout.Length);
            Array.Copy(objects, o, objects.Length);
            return true;
        }
    }

    void Build(byte[] xml)
    {
        var err = new byte[1000];
        IntPtr next = Native.bb_create(xml, xml.Length, err, err.Length);
        if (next == IntPtr.Zero)
        {
            Error = Encoding.UTF8.GetString(err).TrimEnd('\0');
            return;
        }
        if (world != IntPtr.Zero) Native.bb_free(world);
        world = next;
        int n = Native.bb_ngeom(world);
        var geoms = new Geom[n];
        for (int g = 0; g < n; g++)
        {
            var geom = geoms[g] = new Geom();
            Native.bb_geom(world, g, out geom.type, out geom.role, out geom.hand, out geom.finger, out geom.obj, geom.size);
            geom.surface = Native.bb_geom_surface(world, g);
            geom.friction = (float)Native.bb_geom_friction(world, g);
            int nv = Native.bb_mesh(world, g, null, null, out int nf);
            if (nv > 0)
            {
                geom.vert = new float[3 * nv];
                geom.face = new int[3 * nf];
                Native.bb_mesh(world, g, geom.vert, geom.face, out nf);
            }
        }
        Geoms = geoms;
        ObjectCount = Native.bb_nobjects(world);
        Error = null;
        pos = new float[3 * n];
        mat = new float[9 * n];
        objects = new double[7 * ObjectCount];
        foreach (var i in inputs) i.teleport = true;
        Version++;
    }

    void Run()
    {
        var clock = Stopwatch.StartNew();
        double last = clock.Elapsed.TotalSeconds, busy = 0, simmed = 0;
        var r = new double[Native.ReadoutSize];
        var op = new double[3];
        var oq = new double[4];
        float[] p = new float[0], m = new float[0];
        double[] o = new double[0];
        while (running)
        {
            double now = clock.Elapsed.TotalSeconds;
            double dt = Math.Min(now - last, MaxStepS);
            last = now;
            lock (gate)
            {
                if (xmlStaged != null)
                {
                    Build(xmlStaged);
                    xmlStaged = null;
                    if (world != IntPtr.Zero) afterBuild?.Invoke(world);
                    p = new float[pos.Length];
                    m = new float[mat.Length];
                    o = new double[objects.Length];
                }
                if (world != IntPtr.Zero)
                    for (int h = 0; h < 2; h++)
                    {
                        var i = inputs[h];
                        if (i.placed) Native.bb_set_hand(world, h, i.pos, i.quat, i.teleport ? 1 : 0);
                        if (i.placed) i.teleport = false;
                        Native.bb_set_input(world, h, i.angles, i.effort, i.locked ? 1 : 0, i.grab ? 1 : 0);
                    }
            }
            if (world == IntPtr.Zero)
            {
                while (commands.TryDequeue(out _)) { }
                Thread.Sleep(20);
                continue;
            }
            while (commands.TryDequeue(out var command)) command(world);
            double t0 = clock.Elapsed.TotalSeconds;
            Native.bb_step(world, dt);
            Native.bb_poses(world, p, m);
            Native.bb_readout(world, r);
            for (int k = 0; k < o.Length / 7; k++)
            {
                Native.bb_object_pose(world, k, op, oq);
                Array.Copy(op, 0, o, 7 * k, 3);
                Array.Copy(oq, 0, o, 7 * k + 3, 4);
            }
            busy += clock.Elapsed.TotalSeconds - t0;
            simmed += dt;
            lock (gate)
            {
                Array.Copy(p, pos, p.Length);
                Array.Copy(m, mat, m.Length);
                Array.Copy(r, readout, r.Length);
                Array.Copy(o, objects, o.Length);
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
