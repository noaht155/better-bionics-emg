using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;

// The contact sim in the headset: the table in front of the seated user, the hand moved with the right controller's
// grip, settings on the panel, the control layer's output from the laptop when it's plugged in.
// Right trigger: panel buttons. Right grip: hold and move the hand. Left X: recentre.
public class App : MonoBehaviour
{
    public Material material;
    public Transform head;

    // Where the table's centre goes from the head on recentre (m)
    const float TableAhead = 0.4f, TableBelow = 0.45f;
    // clip_workspace in app.py
    const float ReachM = 0.45f;

    [Serializable]
    class Defaults
    {
        public string @object;
        public float mass_scale;
        public float max_force;
        public bool assist;
        public string[] objects;
        public float[] rest;
    }

    [Serializable]
    class ParamsMessage
    {
        public Link.Params @params;
    }

    [Serializable]
    class Readout
    {
        public float grip_force;
        public float[] finger_force;
        public float palm_force;
        public float object_height;
        public float speed;
        public int blowups;
    }

    [Serializable]
    class ReadoutMessage
    {
        public Readout readout;
    }

    Transform root, left, right, ray;
    Sim sim;
    SimView view;
    Link link;
    Panel panel;
    Defaults defaults;
    readonly Dictionary<string, string> models = new Dictionary<string, string>();
    Link.Params current;
    int seq = -1;
    TextMesh objectText, massText, forceText, assistText, linkText, gripText;
    readonly double[] readout = new double[Sim.ReadoutSize], hand = new double[7];
    readonly double[] handPos = new double[3], handQuat = new double[4];
    bool dragging, prevTrigger, prevX, centred;
    Vector3 grabPos, grabHandPos;
    Quaternion grabRot, grabHandRot;
    float headSince = -1;
    readonly List<XRInputSubsystem> inputs = new List<XRInputSubsystem>();
    bool floorSet;

    void Awake()
    {
        Application.targetFrameRate = 90;
        defaults = JsonUtility.FromJson<Defaults>(Resources.Load<TextAsset>("sim/defaults").text);
        foreach (var obj in defaults.objects) models[obj] = Resources.Load<TextAsset>("sim/" + obj).text;
        current = new Link.Params
        {
            @object = defaults.@object, mass_scale = defaults.mass_scale, max_force = defaults.max_force,
            assist = defaults.assist
        };

        root = new GameObject("table root").transform;
        left = Marker("left controller");
        right = Marker("right controller");
        ray = GameObject.CreatePrimitive(PrimitiveType.Cube).transform;
        Destroy(ray.GetComponent<Collider>());
        ray.GetComponent<Renderer>().sharedMaterial = new Material(material) { color = SimView.Hex(0xff3d8b) };
        ray.gameObject.SetActive(false);
        var floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
        Destroy(floor.GetComponent<Collider>());
        floor.GetComponent<Renderer>().sharedMaterial = new Material(material) { color = SimView.Hex(0x0a0a0b) };

        view = new SimView(root, material);
        BuildPanel();
        sim = new Sim();
        sim.Load(current.@object, models[current.@object]);
        ApplyParams();
        sim.Start();
        link = new Link();
        link.Start();
    }

    Transform Marker(string name)
    {
        var t = GameObject.CreatePrimitive(PrimitiveType.Cube).transform;
        t.name = name;
        t.localScale = new Vector3(0.03f, 0.03f, 0.08f);
        Destroy(t.GetComponent<Collider>());
        t.GetComponent<Renderer>().sharedMaterial = new Material(material) { color = SimView.Hex(0x3f3f46) };
        return t;
    }

    void BuildPanel()
    {
        panel = new Panel(root, material);
        // Left of the table, turned towards where the head is after a recentre
        panel.root.localPosition = new Vector3(-0.42f, 0.32f, 0.05f);
        panel.root.localRotation = Quaternion.LookRotation(panel.root.localPosition - new Vector3(0, TableBelow, -TableAhead));
        objectText = panel.Row("", ("<", () => StepObject(-1)), (">", () => StepObject(1)));
        massText = panel.Row("", ("-", () => Change(p => p.mass_scale = Mathf.Clamp(p.mass_scale - 0.25f, 0.25f, 3))),
            ("+", () => Change(p => p.mass_scale = Mathf.Clamp(p.mass_scale + 0.25f, 0.25f, 3))));
        forceText = panel.Row("", ("-", () => Change(p => p.max_force = Mathf.Clamp(p.max_force - 1, 2, 40))),
            ("+", () => Change(p => p.max_force = Mathf.Clamp(p.max_force + 1, 2, 40))));
        assistText = panel.Row("", ("on/off", () => Change(p => p.assist = !p.assist)));
        panel.Row("", ("Reset object", () => sim.ResetObject()), ("Recentre", Recentre));
        linkText = panel.Row("");
        gripText = panel.Row("");
        panel.Finish();
    }

    void StepObject(int by)
    {
        int i = Array.IndexOf(defaults.objects, current.@object);
        int n = defaults.objects.Length;
        Change(p => p.@object = defaults.objects[((i + by) % n + n) % n]);
    }

    // A change made here goes to the sim and to the laptop, which shares it with its own page
    void Change(Action<Link.Params> edit)
    {
        string before = current.@object;
        edit(current);
        if (current.@object != before) sim.Load(current.@object, models[current.@object]);
        ApplyParams();
        link.Send(JsonUtility.ToJson(new ParamsMessage { @params = current }));
    }

    void ApplyParams()
    {
        sim.SetParams(current.mass_scale, current.max_force, current.assist);
    }

    // Settings changed on the laptop (its params_seq went up) replace the ones here
    void TakeLaptopParams(Link.Frame f)
    {
        if (f.@params == null || f.@params.@object == null || f.params_seq == seq) return;
        seq = f.params_seq;
        var p = f.@params;
        if (p.@object != current.@object && models.ContainsKey(p.@object)) sim.Load(p.@object, models[p.@object]);
        if (models.ContainsKey(p.@object)) current.@object = p.@object;
        current.mass_scale = p.mass_scale;
        current.max_force = p.max_force;
        current.assist = p.assist;
        ApplyParams();
    }

    static bool Pose(XRNode node, out Vector3 p, out Quaternion q, bool eye = false)
    {
        var d = InputDevices.GetDeviceAtXRNode(node);
        p = default;
        q = Quaternion.identity;
        return d.isValid
               && d.TryGetFeatureValue(eye ? CommonUsages.centerEyePosition : CommonUsages.devicePosition, out p)
               && d.TryGetFeatureValue(eye ? CommonUsages.centerEyeRotation : CommonUsages.deviceRotation, out q);
    }

    static bool Button(XRNode node, InputFeatureUsage<bool> usage)
    {
        var d = InputDevices.GetDeviceAtXRNode(node);
        return d.isValid && d.TryGetFeatureValue(usage, out bool on) && on;
    }

    void OnEnable() => Application.onBeforeRender += TrackHead;

    void OnDisable() => Application.onBeforeRender -= TrackHead;

    void TrackHead()
    {
        if (Pose(XRNode.Head, out var p, out var q, true)) head.SetPositionAndRotation(p, q);
    }

    void Recentre()
    {
        var forward = Vector3.ProjectOnPlane(head.forward, Vector3.up);
        if (forward.sqrMagnitude < 1e-4f) forward = Vector3.forward;
        var yaw = Quaternion.LookRotation(forward.normalized, Vector3.up);
        root.SetPositionAndRotation(head.position + yaw * new Vector3(0, -TableBelow, TableAhead), yaw);
    }

    void Update()
    {
        if (!floorSet)
        {
            SubsystemManager.GetSubsystems(inputs);
            foreach (var s in inputs) floorSet |= s.TrySetTrackingOriginMode(TrackingOriginModeFlags.Floor);
        }
        TrackHead();
        // Recentre once the head has been tracked for a moment, the first poses after start can be zero
        if (!centred && head.position != Vector3.zero)
        {
            if (headSince < 0) headSince = Time.time;
            if (Time.time - headSince > 0.5f) { Recentre(); centred = true; }
        }
        bool hasLeft = Pose(XRNode.LeftHand, out var lp, out var lq);
        bool hasRight = Pose(XRNode.RightHand, out var rp, out var rq);
        left.gameObject.SetActive(hasLeft);
        right.gameObject.SetActive(hasRight);
        if (hasLeft) left.SetPositionAndRotation(lp, lq);
        if (hasRight) right.SetPositionAndRotation(rp, rq);

        bool x = Button(XRNode.LeftHand, CommonUsages.primaryButton);
        if (x && !prevX) Recentre();
        prevX = x;

        var frame = link.Latest;
        if (frame != null)
        {
            TakeLaptopParams(frame);
            sim.SetInputs(frame.angles, frame.effort, frame.locked);
        }
        else
        {
            sim.SetInputs(defaults.rest, 0, false);
        }

        bool viewed = view.Update(sim, readout, hand);
        if (viewed && hasRight) DragHand(rp, rq);

        bool trigger = Button(XRNode.RightHand, CommonUsages.triggerButton);
        float length = hasRight ? panel.Point(new Ray(rp, rq * Vector3.forward), trigger && !prevTrigger) : 0;
        prevTrigger = trigger;
        ray.gameObject.SetActive(length > 0);
        if (length > 0)
        {
            ray.SetPositionAndRotation(rp + rq * Vector3.forward * (length / 2), rq);
            ray.localScale = new Vector3(0.002f, 0.002f, length);
        }

        ShowLabels(frame);
        if (viewed) SendReadout();
    }

    // While the grip is held the hand follows the controller's movement from where it was grabbed
    void DragHand(Vector3 cp, Quaternion cq)
    {
        bool grip = Button(XRNode.RightHand, CommonUsages.gripButton);
        if (!grip)
        {
            dragging = false;
            return;
        }
        Vector3 localC = root.InverseTransformPoint(cp);
        Quaternion localQ = Quaternion.Inverse(root.rotation) * cq;
        if (!dragging)
        {
            dragging = true;
            grabPos = localC;
            grabRot = localQ;
            grabHandPos = SimView.ToUnity(hand[0], hand[1], hand[2]);
            grabHandRot = SimView.QuatToUnity(hand, 3);
        }
        var turn = localQ * Quaternion.Inverse(grabRot);
        var target = localC + turn * (grabHandPos - grabPos);
        var rot = turn * grabHandRot;
        SimView.ToSim(target, handPos);
        handPos[0] = Mathf.Clamp((float)handPos[0], -ReachM, ReachM);
        handPos[1] = Mathf.Clamp((float)handPos[1], -ReachM, ReachM);
        handPos[2] = Mathf.Clamp((float)handPos[2], 0, ReachM);
        SimView.QuatToSim(rot, handQuat);
        sim.SetHand(handPos, handQuat);
    }

    void ShowLabels(Link.Frame frame)
    {
        objectText.text = "Object: " + current.@object;
        massText.text = $"Mass x {current.mass_scale:0.00}";
        forceText.text = $"Hand strength {current.max_force:0} N";
        assistText.text = "Grasp assist: " + (current.assist ? "on" : "off");
        linkText.text = !link.Connected ? "Laptop: not connected, hand at rest"
            : frame != null && frame.driven ? (frame.locked ? "Laptop: control, locked" : "Laptop: control")
            : "Laptop: connected, no control running";
        gripText.text = sim.Error ?? $"Grip {readout[0]:0.0} N, sim {sim.Speed:0.0}x real time";
    }

    void SendReadout()
    {
        var r = new Readout
        {
            grip_force = (float)readout[0], finger_force = new float[5], palm_force = (float)readout[6],
            object_height = (float)readout[10], speed = (float)sim.Speed, blowups = (int)readout[12]
        };
        for (int i = 0; i < 5; i++) r.finger_force[i] = (float)readout[1 + i];
        link.Readout = JsonUtility.ToJson(new ReadoutMessage { readout = r });
    }

    void OnDestroy()
    {
        sim?.Stop();
        link?.Stop();
    }
}
