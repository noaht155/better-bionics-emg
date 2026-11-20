using System;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.XR;

// The contact sim in the headset. Both controllers are hands: the one strapped to the prosthetic arm places the
// prosthetic hand, whose fingers follow the control layer's output from the laptop; the other is the free hand on
// the normal controls (grip closes the fingers and holds what it touches, trigger curls the index, thumb rests on
// the buttons). The free hand's B/Y opens the menu, its ray and trigger work it.
public class App : MonoBehaviour
{
    public Material material, uiMaterial;
    public Transform head;

    const float GrabOn = 0.6f, GrabOff = 0.35f;
    // The desk's near edge this far in front of the eyes when it's put in front of the user
    const float DeskAhead = 0.4f;

    [Serializable]
    class Meta
    {
        public string @object;
        public float mass_scale, max_force;
        public bool assist;
        public string[] objects, pool, joints;
        public float[] half, rest, fist, point, desk_half, park;
        public float desk_centre_y, desk_height;
    }

    [Serializable]
    class ParamsMessage
    {
        public Link.Params @params;
    }

    [Serializable]
    class Readout
    {
        public float grip_force, palm_force, free_grip_force, speed;
        public float[] finger_force;
        public int blowups;
    }

    [Serializable]
    class ReadoutMessage
    {
        public Readout readout;
    }

    // For the laptop's VR tab, in sim coordinates: every geom once, then the poses and the head
    [Serializable]
    class SceneGeom
    {
        public string type, role;
        public int hand, finger, obj;
        public float[] size, vertices;
        public int[] faces;
    }

    [Serializable]
    class SceneMessage
    {
        public SceneBody scene;
    }

    [Serializable]
    class SceneBody
    {
        public int version;
        public bool prosthetic_left;
        public SceneGeom[] geoms;
    }

    [Serializable]
    class View
    {
        public float[] pos, mat, head;
        public bool[] parked;
        public int[] held;
        public bool menu;
    }

    [Serializable]
    class ViewMessage
    {
        public View view;
    }

    static readonly string[] GeomTypes = { "plane", "hfield", "sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh" };

    Meta meta;
    Settings settings;
    string sceneLeft, sceneRight;
    Transform deskRoot, ray;
    readonly Transform[] controllerModels = new Transform[2];
    Room room;
    Sim sim;
    SimView view;
    Link link;
    Ui ui;
    Ui.Page setupPage;
    Link.Params current;
    int seq = -1, blowups, sceneSent = -1, lastCommand = -1;
    bool wasConnected;
    readonly View viewOut = new View { head = new float[9], held = new int[2] };
    readonly double[] headSim = new double[3];
    string laptopObject;
    bool[] parked;
    int[] fingerOf;
    readonly double[] readout = new double[Native.ReadoutSize];
    readonly double[] simPos = new double[3], simQuat = new double[4];
    readonly float[] freeAngles = new float[Sim.Joints];
    bool grabbing, prevMenu, prevTrigger, movingDesk, deskDragging, adjustFree, showControllers;
    int heldBefore = -1;
    Vector3 dragStart, deskStart;
    float dragYaw, deskYawStart;
    readonly List<XRInputSubsystem> inputs = new List<XRInputSubsystem>();
    bool floorSet;
    TextMeshPro linkText, simText, massText, forceText, assistText, heightText, sideText, adjustText, offsetText;

    XRNode ProstheticNode => settings.prostheticLeft ? XRNode.LeftHand : XRNode.RightHand;
    XRNode FreeNode => settings.prostheticLeft ? XRNode.RightHand : XRNode.LeftHand;

    void Awake()
    {
        Application.targetFrameRate = 90;
        meta = JsonUtility.FromJson<Meta>(Resources.Load<TextAsset>("sim/scene").text);
        sceneLeft = Resources.Load<TextAsset>("sim/scene_left").text;
        sceneRight = Resources.Load<TextAsset>("sim/scene_right").text;
        settings = Settings.Load();
        current = new Link.Params
        {
            @object = meta.@object, mass_scale = meta.mass_scale, max_force = meta.max_force, assist = meta.assist
        };
        fingerOf = new int[Sim.Joints];
        string[] fingers = { "thumb", "index", "middle", "ring", "pinky" };
        for (int j = 0; j < Sim.Joints; j++) fingerOf[j] = Array.IndexOf(fingers, meta.joints[j].Split('_')[0]);

        deskRoot = new GameObject("desk root").transform;
        room = new Room(deskRoot, material, new Vector2(meta.desk_half[0], meta.desk_half[1]), meta.desk_centre_y,
            2 * meta.desk_half[2]);
        room.SetHeight(settings.deskHeight);
        PlaceDeskFromSettings();
        view = new SimView(deskRoot, material);
        for (int i = 0; i < 2; i++)
        {
            controllerModels[i] = GameObject.CreatePrimitive(PrimitiveType.Cube).transform;
            Destroy(controllerModels[i].GetComponent<Collider>());
            controllerModels[i].localScale = new Vector3(0.04f, 0.04f, 0.1f);
            controllerModels[i].GetComponent<Renderer>().sharedMaterial = SimView.Paint(material, Ui.LineStrong);
        }
        ray = GameObject.CreatePrimitive(PrimitiveType.Cube).transform;
        Destroy(ray.GetComponent<Collider>());
        ray.GetComponent<Renderer>().sharedMaterial = new Material(uiMaterial) { color = Ui.Accent };
        ray.gameObject.SetActive(false);
        BuildMenu();

        sim = new Sim();
        LoadScene();
        sim.Start();
        link = new Link();
        link.Start();
    }

    void LoadScene()
    {
        parked = new bool[meta.pool.Length];
        for (int i = 0; i < parked.Length; i++) parked[i] = true;
        float height = settings.deskHeight;
        var p = current;
        // Set up inside the build, so nothing queued for the old scene lands on the new one
        var first = SpawnSpot(current.@object);
        sim.Load(settings.prostheticLeft ? sceneLeft : sceneRight, w =>
        {
            Native.bb_set_floor(w, height);
            Native.bb_set_params(w, p.mass_scale, p.max_force, p.assist ? 1 : 0);
            if (first.i >= 0) Native.bb_place_object(w, first.i, first.pos, new double[] { 1, 0, 0, 0 }, 0);
        });
        if (first.i >= 0) parked[first.i] = false;
    }

    // Menu

    void BuildMenu()
    {
        ui = new Ui(uiMaterial);
        var play = ui.AddPage("Play");
        ui.Heading(play, "STATUS");
        linkText = ui.Row(play, "");
        simText = ui.Row(play, "");
        ui.Heading(play, "PHYSICS");
        massText = ui.Row(play, "", ("-", () => Change(p => p.mass_scale = Mathf.Clamp(p.mass_scale - 0.25f, 0.25f, 3))),
            ("+", () => Change(p => p.mass_scale = Mathf.Clamp(p.mass_scale + 0.25f, 0.25f, 3))));
        forceText = ui.Row(play, "", ("-", () => Change(p => p.max_force = Mathf.Clamp(p.max_force - 1, 2, 40))),
            ("+", () => Change(p => p.max_force = Mathf.Clamp(p.max_force + 1, 2, 40))));
        assistText = ui.Row(play, "Grasp assist", () => current.assist, ("On", () => Change(p => p.assist = !p.assist)));

        var objects = ui.AddPage("Objects");
        ui.Heading(objects, "SPAWN ON THE DESK");
        var spawns = new List<(string, Action)>();
        foreach (var kind in meta.objects)
        {
            var k = kind;
            spawns.Add((Title(kind), () => Spawn(k)));
        }
        ui.Buttons(objects, spawns.ToArray());
        ui.Buttons(objects, ("Clear the desk", ClearDesk));
        ui.Heading(objects, "DESK");
        ui.Row(objects, "Move the desk with your grip", () => movingDesk, ("Move", () => movingDesk = !movingDesk));
        heightText = ui.Row(objects, "", ("-", () => SetDeskHeight(settings.deskHeight - 0.01f)),
            ("+", () => SetDeskHeight(settings.deskHeight + 0.01f)));
        ui.Buttons(objects, ("Put the desk in front of me", DeskToMe));

        setupPage = ui.AddPage("Setup");
        ui.Heading(setupPage, "PROSTHETIC ARM");
        sideText = ui.Row(setupPage, "", ("Left", () => SetSide(true)), ("Right", () => SetSide(false)));
        ui.Heading(setupPage, "HAND OFFSET FROM ITS CONTROLLER");
        adjustText = ui.Row(setupPage, "", ("Prosthetic", () => adjustFree = false), ("Free", () => adjustFree = true));
        ui.Row(setupPage, "Forward", ("-1 cm", () => Nudge(new Vector3(0, 0, -0.01f), Vector3.zero)),
            ("+1 cm", () => Nudge(new Vector3(0, 0, 0.01f), Vector3.zero)));
        ui.Row(setupPage, "Up", ("-1 cm", () => Nudge(new Vector3(0, -0.01f, 0), Vector3.zero)),
            ("+1 cm", () => Nudge(new Vector3(0, 0.01f, 0), Vector3.zero)));
        ui.Row(setupPage, "Right", ("-1 cm", () => Nudge(new Vector3(-0.01f, 0, 0), Vector3.zero)),
            ("+1 cm", () => Nudge(new Vector3(0.01f, 0, 0), Vector3.zero)));
        ui.Row(setupPage, "Tilt", ("-5°", () => Nudge(Vector3.zero, new Vector3(-5, 0, 0))),
            ("+5°", () => Nudge(Vector3.zero, new Vector3(5, 0, 0))));
        ui.Row(setupPage, "Turn", ("-5°", () => Nudge(Vector3.zero, new Vector3(0, -5, 0))),
            ("+5°", () => Nudge(Vector3.zero, new Vector3(0, 5, 0))));
        ui.Row(setupPage, "Roll", ("-5°", () => Nudge(Vector3.zero, new Vector3(0, 0, -5))),
            ("+5°", () => Nudge(Vector3.zero, new Vector3(0, 0, 5))));
        offsetText = ui.Note(setupPage);
        ui.Buttons(setupPage, ("Reset this hand", ResetOffset), ("Show controllers", () => showControllers = !showControllers));
        ui.Show(play);
        ui.Close();
    }

    static string Title(string s) => s == "small cube" ? "Cube" : char.ToUpper(s[0]) + s.Substring(1);

    Settings.Offset Adjusting => adjustFree ? settings.free : settings.prosthetic;

    void Nudge(Vector3 pos, Vector3 rot)
    {
        Adjusting.pos += pos;
        Adjusting.rot += rot;
        settings.Save();
    }

    void ResetOffset()
    {
        Adjusting.pos = Settings.HeldPos;
        Adjusting.rot = Vector3.zero;
        settings.Save();
    }

    void SetSide(bool left)
    {
        if (settings.prostheticLeft == left) return;
        settings.prostheticLeft = left;
        settings.Save();
        LoadScene();
    }

    // A change made here goes to the sim and to the laptop, which shares it with its own page
    void Change(Action<Link.Params> edit)
    {
        edit(current);
        ApplyParams();
        link.Send(JsonUtility.ToJson(new ParamsMessage { @params = current }));
    }

    void ApplyParams()
    {
        var p = current;
        sim.Do(w => Native.bb_set_params(w, p.mass_scale, p.max_force, p.assist ? 1 : 0));
    }

    // Settings changed on the laptop (its params_seq went up) replace the ones here; another object there clears the
    // desk and puts that one on it
    void TakeLaptopParams(Link.Frame f)
    {
        var p = f.@params;
        if (p == null || p.@object == null || f.params_seq == seq) return;
        seq = f.params_seq;
        current.mass_scale = p.mass_scale;
        current.max_force = p.max_force;
        current.assist = p.assist;
        ApplyParams();
        if (laptopObject != null && p.@object != laptopObject && Array.IndexOf(meta.objects, p.@object) >= 0)
        {
            ClearDesk();
            Spawn(p.@object);
        }
        laptopObject = p.@object;
        current.@object = p.@object;
    }

    // Objects

    double[] ParkPos(int i)
    {
        int kind = Array.IndexOf(meta.objects, meta.pool[i]);
        return new double[] { meta.park[0] + meta.park[2] * i, meta.park[1], -settings.deskHeight + meta.half[kind] + 0.001 };
    }

    void Park(int i)
    {
        parked[i] = true;
        var pos = ParkPos(i);
        sim.Do(w => Native.bb_place_object(w, i, pos, new double[] { 1, 0, 0, 0 }, 1));
    }

    void ClearDesk()
    {
        for (int i = 0; i < parked.Length; i++) Park(i);
    }

    // A parked object of this kind and a free spot for it on the desk; i is -1 when every one of them is out
    (int i, double[] pos) SpawnSpot(string kind)
    {
        int i = Array.FindIndex(meta.pool, k => k == kind);
        while (i >= 0 && i < parked.Length && !(parked[i] && meta.pool[i] == kind)) i++;
        if (i < 0 || i >= parked.Length || meta.pool[i] != kind) return (-1, null);
        int onDesk = 0;
        foreach (bool p in parked) if (!p) onDesk++;
        float[] spots = { 0, 0.14f, -0.14f, 0.28f, -0.28f, 0.07f, -0.07f, 0.21f, -0.21f, 0.35f };
        float half = meta.half[Array.IndexOf(meta.objects, kind)];
        return (i, new double[] { spots[onDesk % spots.Length], -0.12, half + 0.003 });
    }

    void Spawn(string kind)
    {
        var (i, pos) = SpawnSpot(kind);
        if (i < 0) return;
        parked[i] = false;
        sim.Do(w => Native.bb_place_object(w, i, pos, new double[] { 1, 0, 0, 0 }, 0));
    }

    // Desk

    void PlaceDeskFromSettings()
    {
        deskRoot.SetPositionAndRotation(new Vector3(settings.deskPos.x, settings.deskHeight, settings.deskPos.z),
            Quaternion.Euler(0, settings.deskYaw, 0));
    }

    void DeskToMe()
    {
        var forward = Vector3.ProjectOnPlane(head.forward, Vector3.up);
        if (forward.sqrMagnitude < 1e-4f) forward = Vector3.forward;
        var yaw = Quaternion.LookRotation(forward.normalized, Vector3.up);
        settings.deskPos = head.position + yaw * new Vector3(0, 0, DeskAhead);
        settings.deskYaw = yaw.eulerAngles.y;
        settings.deskPlaced = true;
        settings.Save();
        PlaceDeskFromSettings();
    }

    void SetDeskHeight(float height)
    {
        settings.deskHeight = Mathf.Clamp(height, 0.5f, 1.1f);
        settings.Save();
        room.SetHeight(settings.deskHeight);
        PlaceDeskFromSettings();
        float h = settings.deskHeight;
        sim.Do(w => Native.bb_set_floor(w, h));
        var objects = view.Objects;
        for (int i = 0; i < parked.Length; i++)
            if (parked[i] || (7 * i + 2 < objects.Length && objects[7 * i + 2] < -0.05)) Park(i);
    }

    void DragDesk(Vector3 cp, Quaternion cq, bool grip)
    {
        if (!grip)
        {
            if (deskDragging) settings.Save();
            deskDragging = false;
            return;
        }
        float yaw = cq.eulerAngles.y;
        if (!deskDragging)
        {
            deskDragging = true;
            dragStart = cp;
            dragYaw = yaw;
            deskStart = settings.deskPos;
            deskYawStart = settings.deskYaw;
        }
        // Turns about where the grip started and moves with the hand, level: the height is set on the menu
        float turned = Mathf.DeltaAngle(dragYaw, yaw);
        var moved = Vector3.ProjectOnPlane(cp - dragStart, Vector3.up);
        settings.deskPos = dragStart + moved + Quaternion.Euler(0, turned, 0) * (deskStart - dragStart);
        settings.deskYaw = deskYawStart + turned;
        PlaceDeskFromSettings();
    }

    // Tracking

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

    static float Axis(XRNode node, InputFeatureUsage<float> usage)
    {
        var d = InputDevices.GetDeviceAtXRNode(node);
        return d.isValid && d.TryGetFeatureValue(usage, out float v) ? v : 0;
    }

    static void Buzz(XRNode node, float amount, float seconds) =>
        InputDevices.GetDeviceAtXRNode(node).SendHapticImpulse(0, amount, seconds);

    void OnEnable() => Application.onBeforeRender += TrackHead;

    void OnDisable() => Application.onBeforeRender -= TrackHead;

    void TrackHead()
    {
        if (Pose(XRNode.Head, out var p, out var q, true)) head.SetPositionAndRotation(p, q);
    }

    // The wrist pose for a hand on its controller: the hand as it holds a controller (fingers forward, palm towards
    // the controller's middle, thumb up), turned and moved by the setup's offset
    static void HandPose(Vector3 cp, Quaternion cq, bool leftHand, Settings.Offset offset, out Vector3 p, out Quaternion q)
    {
        var held = Quaternion.LookRotation(Vector3.back, leftHand ? Vector3.right : Vector3.left);
        q = cq * Quaternion.Euler(offset.rot) * held;
        p = cp + cq * offset.pos;
    }

    void PlaceHand(int h, Vector3 p, Quaternion q)
    {
        SimView.ToSim(deskRoot.InverseTransformPoint(p), simPos);
        SimView.QuatToSim(Quaternion.Inverse(deskRoot.rotation) * q, simQuat);
        sim.SetHand(h, simPos, simQuat);
    }

    void Update()
    {
        if (!floorSet)
        {
            SubsystemManager.GetSubsystems(inputs);
            foreach (var s in inputs) floorSet |= s.TrySetTrackingOriginMode(TrackingOriginModeFlags.Floor);
        }
        TrackHead();
        if (!settings.deskPlaced && head.position.y > 0.3f) DeskToMe();

        bool hasP = Pose(ProstheticNode, out var pp, out var pq);
        bool hasF = Pose(FreeNode, out var fp, out var fq);
        controllerModels[0].gameObject.SetActive(hasP && showControllers);
        controllerModels[1].gameObject.SetActive(hasF && showControllers);
        if (hasP) controllerModels[0].SetPositionAndRotation(pp, pq);
        if (hasF) controllerModels[1].SetPositionAndRotation(fp, fq);
        if (hasP)
        {
            HandPose(pp, pq, settings.prostheticLeft, settings.prosthetic, out var p, out var q);
            PlaceHand(Sim.Prosthetic, p, q);
        }
        if (hasF)
        {
            HandPose(fp, fq, !settings.prostheticLeft, settings.free, out var p, out var q);
            PlaceHand(Sim.Free, p, q);
        }

        var frame = link.Latest;
        if (frame != null)
        {
            TakeLaptopParams(frame);
            RunCommands(frame);
            sim.SetInput(Sim.Prosthetic, frame.angles, frame.effort, frame.locked, false);
        }
        else
        {
            sim.SetInput(Sim.Prosthetic, meta.rest, 0, false, false);
        }

        bool menu = Button(FreeNode, CommonUsages.secondaryButton);
        if (menu && !prevMenu)
        {
            if (ui.Visible) ui.Close();
            else ui.Open(head);
        }
        prevMenu = menu;

        float grip = Axis(FreeNode, CommonUsages.grip), trigger = Axis(FreeNode, CommonUsages.trigger);
        bool pressed = trigger > 0.7f && !prevTrigger;
        prevTrigger = trigger > 0.7f;
        bool clicked = false;
        float length = hasF ? ui.Point(new Ray(fp, fq * Vector3.forward), pressed, out clicked) : 0;
        if (clicked) Buzz(FreeNode, 0.25f, 0.03f);
        ray.gameObject.SetActive(ui.Visible && hasF);
        if (ui.Visible && hasF)
        {
            float shown = length > 0 ? length : 0.6f;
            ray.SetPositionAndRotation(fp + fq * Vector3.forward * (shown / 2), fq);
            ray.localScale = new Vector3(0.0015f, 0.0015f, shown);
        }

        if (movingDesk && hasF) DragDesk(fp, fq, grip > GrabOn);
        FreeHandInput(grip, trigger, movingDesk || length > 0);

        bool viewed = view.Update(sim, readout, parked);
        if (viewed)
        {
            int held = (int)readout[9];
            if (held >= 0 && heldBefore < 0) Buzz(FreeNode, 0.4f, 0.04f);
            heldBefore = held;
            if ((int)readout[13] != blowups)
            {
                blowups = (int)readout[13];
                ClearDesk();
            }
            SendReadout();
            SendView();
        }
        if (link.Connected && !wasConnected) sceneSent = -1;
        wasConnected = link.Connected;
        if (!link.Connected) lastCommand = -1;
        ShowLabels(frame);
    }

    // Grip closes the middle, ring and little fingers, the trigger the index, the thumb comes down while it rests on
    // the buttons or stick; the grip holds what the hand touches. Pointing at the menu or moving the desk doesn't grab
    void FreeHandInput(float grip, float trigger, bool busy)
    {
        var node = FreeNode;
        bool thumbDown = Button(node, CommonUsages.primaryTouch) || Button(node, CommonUsages.secondaryTouch)
                         || Button(node, CommonUsages.primary2DAxisTouch);
        if (busy) grip = trigger = 0;
        for (int j = 0; j < Sim.Joints; j++)
        {
            int f = fingerOf[j];
            float amount = f == 1 ? trigger : f == 0 ? (thumbDown ? Mathf.Max(grip, 0.5f) : 0.2f * grip) : grip;
            freeAngles[j] = Mathf.Lerp(meta.rest[j], meta.fist[j], amount);
        }
        grabbing = grabbing ? grip > GrabOff : grip > GrabOn;
        sim.SetInput(Sim.Free, freeAngles, Mathf.Max(grip, trigger), false, grabbing);
    }

    void ShowLabels(Link.Frame frame)
    {
        linkText.text = !link.Connected ? "Laptop: not connected, hand at rest"
            : frame != null && frame.driven ? (frame.locked ? "Laptop: control, locked" : "Laptop: control running")
            : "Laptop: connected, no control running";
        simText.text = sim.Error ?? $"Grip {readout[0]:0.0} N   ·   sim {sim.Speed:0.0}x real time";
        massText.text = $"Mass  x{current.mass_scale:0.00}";
        forceText.text = $"Hand strength  {current.max_force:0} N";
        heightText.text = $"Height  {settings.deskHeight * 100:0} cm";
        sideText.text = settings.prostheticLeft ? "Left arm" : "Right arm";
        adjustText.text = adjustFree ? "Adjusting: free hand" : "Adjusting: prosthetic hand";
        var o = Adjusting;
        offsetText.text = $"Wrist {o.pos.z * 100:0} cm forward, {o.pos.y * 100:0} up, {o.pos.x * 100:0} right of the " +
                          $"controller; tilt {o.rot.x:0}°, turn {o.rot.y:0}°, roll {o.rot.z:0}°";
    }

    void SendReadout()
    {
        var r = new Readout
        {
            grip_force = (float)readout[0], finger_force = new float[5], palm_force = (float)readout[6],
            free_grip_force = (float)readout[7], speed = (float)sim.Speed, blowups = (int)readout[13]
        };
        for (int i = 0; i < 5; i++) r.finger_force[i] = (float)readout[1 + i];
        link.Readout = JsonUtility.ToJson(new ReadoutMessage { readout = r });
    }

    // Spawn and clear from the laptop's VR tab. Commands already there when the link comes up are old, skipped
    void RunCommands(Link.Frame f)
    {
        if (f.commands == null) return;
        int newest = lastCommand;
        foreach (var c in f.commands)
        {
            if (lastCommand >= 0 && c.id > lastCommand)
            {
                if (c.kind == "spawn") Spawn(c.arg);
                else if (c.kind == "clear") ClearDesk();
            }
            newest = Math.Max(newest, c.id);
        }
        lastCommand = Math.Max(newest, 0);
    }

    void SendView()
    {
        if (!link.Connected) return;
        if (sceneSent != sim.Version && sim.Geoms != null)
        {
            var geoms = new SceneGeom[sim.Geoms.Length];
            for (int g = 0; g < geoms.Length; g++)
            {
                var src = sim.Geoms[g];
                geoms[g] = new SceneGeom
                {
                    type = GeomTypes[src.type], role = src.role == 1 ? "object" : src.role == 2 ? "hand" : "static",
                    hand = src.hand, finger = src.finger, obj = src.obj,
                    size = new[] { (float)src.size[0], (float)src.size[1], (float)src.size[2] },
                    vertices = src.vert ?? new float[0], faces = src.face ?? new int[0]
                };
            }
            link.Scene = JsonUtility.ToJson(new SceneMessage
            {
                scene = new SceneBody { version = sim.Version, prosthetic_left = settings.prostheticLeft, geoms = geoms }
            });
            sceneSent = sim.Version;
        }
        view.Poses(out viewOut.pos, out viewOut.mat);
        SimView.ToSim(deskRoot.InverseTransformPoint(head.position), headSim);
        for (int k = 0; k < 3; k++) viewOut.head[k] = (float)headSim[k];
        SimView.ToSim(deskRoot.InverseTransformDirection(head.forward), headSim);
        for (int k = 0; k < 3; k++) viewOut.head[3 + k] = (float)headSim[k];
        SimView.ToSim(deskRoot.InverseTransformDirection(head.up), headSim);
        for (int k = 0; k < 3; k++) viewOut.head[6 + k] = (float)headSim[k];
        viewOut.parked = parked;
        viewOut.held[0] = (int)readout[8];
        viewOut.held[1] = (int)readout[9];
        viewOut.menu = ui.Visible;
        link.View = JsonUtility.ToJson(new ViewMessage { view = viewOut });
    }

    void OnApplicationPause(bool paused)
    {
        if (paused) settings.Save();
    }

    void OnDestroy()
    {
        sim?.Stop();
        link?.Stop();
    }
}
