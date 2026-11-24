using System;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.XR;

// The contact sim in the headset. The prosthetic hand is placed by the controller strapped to that arm and its
// fingers follow the control layer from the laptop. The free hand is the real hand, hand tracked (Multimodal.cs),
// fingers and all; a controller held in it works too (grip, trigger, thumb). The menu is touched with the fingertip
// or pointed at with a controller, and opens from the button by the desk or B/Y. Recording sessions run from the
// menu, with each cue on a card over the desk and its target pose on a ghost hand.
public class App : MonoBehaviour
{
    public Material material, uiMaterial;
    public Transform head;
    public Camera eyes;
    // ARFoundation's camera manager on the head camera: enabled, the headset shows passthrough
    public Behaviour passthroughCamera;
    public Font interFont, interBoldFont, monoFont;

    const float GrabOn = 0.6f, GrabOff = 0.35f;
    const float DeskAhead = 0.4f;
    // A fist this close to the prosthetic hand grabs it while adjusting (m)
    const float AdjustReach = 0.12f;
    // The ghost hand comes in from rest to the cue's pose over this long, as the page's target hand does
    const float FollowInS = 1.0f;
    // The prosthetic palm this far from the controller's hand pose (m, readout 17) shows a ghost hand there, so it is
    // clear the desk or an object holds the hand back; hidden again under ReachHideM. Moving fast, the palm trails by
    // about 0.04 s times the speed, so under ReachShowM for normal moves
    const float ReachShowM = 0.05f, ReachHideM = 0.03f;

    [Serializable]
    class Meta
    {
        public string @object;
        public float mass_scale, max_force;
        public bool assist;
        public string[] objects, pool, joints;
        public float[] half, rest, fist, point, desk_half, park, surface_friction;
        public string rail;
        public float desk_centre_y, desk_height;
    }

    [Serializable] class ParamsMessage { public Link.Params @params; }

    [Serializable]
    class Readout
    {
        public float grip_force, palm_force, free_grip_force, speed, touch_force, touch_friction;
        public float[] finger_force;
        public int blowups, touch_surface;
    }

    [Serializable] class ReadoutMessage { public Readout readout; }

    [Serializable]
    class SceneGeom
    {
        public string type, role, surface;
        public float friction;
        public int hand, finger, obj;
        public float[] size, vertices;
        public int[] faces;
    }

    [Serializable] class SceneMessage { public SceneBody scene; }

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

    [Serializable] class ViewMessage { public View view; }

    [Serializable]
    class RecordAction
    {
        public string action;
        public bool quick;
    }

    [Serializable] class RecordMessage { public RecordAction record; }

    [Serializable] class TrainAction { public string action; }

    [Serializable] class TrainMessage { public TrainAction train; }

    static readonly string[] GeomTypes = { "plane", "hfield", "sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh" };
    // sim.SURFACES
    static readonly string[] SurfaceNames = { "slippery", "smooth", "normal", "rough" };

    Meta meta;
    Settings settings;
    string ghostLeft, ghostRight;
    Transform deskRoot, ray, grabber;
    readonly Transform[] controllerModels = new Transform[2];
    Room room;
    Sim sim;
    SimView view;
    Link link;
    Ui menu, card, dock;
    Ghost ghost, reach;
    TrackedHand tracked;
    bool prevMenuGesture;
    Link.Params current;
    Link.Record record;
    int seq = -1, blowups, sceneSent = -1, lastCommand = -1, heldBefore = -1, cueIndex = -1;
    float cueStartedAt;
    string laptopObject;
    bool[] parked;
    int[] fingerOf;
    readonly double[] readout = new double[Native.ReadoutSize];
    readonly double[] simPos = new double[3], simQuat = new double[4], headSim = new double[3];
    readonly float[] freeAngles = new float[Sim.Joints], ghostAngles = new float[Sim.Joints];
    readonly View viewOut = new View { head = new float[9], held = new int[2] };
    bool grabbing, prevMenuButton, movingDesk, deskDragging, adjusting, adjustDragging, showControllers, wasConnected;
    Vector3 dragStart, deskStart, adjustOffsetPos;
    Quaternion adjustOffsetRot;
    float dragYaw, deskYawStart;
    readonly List<XRInputSubsystem> inputs = new List<XRInputSubsystem>();
    bool floorSet;

    TextMeshPro linkText, simText, hapticsText, trainText, trainError, massText, forceText, heightText, sideText, freeText, recordText, recordError;
    TextMeshPro[] readouts, sessionButtons;
    TextMeshPro cueMeta, cueTitle, cueBody, cueNext, cueState;
    Action<float> cueProgress;

    XRNode ProstheticNode => settings.prostheticLeft ? XRNode.LeftHand : XRNode.RightHand;
    XRNode FreeNode => settings.prostheticLeft ? XRNode.RightHand : XRNode.LeftHand;
    bool FreeIsRight => settings.prostheticLeft;

    void Awake()
    {
        Application.targetFrameRate = 90;
        Ui.InterFont = interFont;
        Ui.InterBoldFont = interBoldFont;
        Ui.MonoFont = monoFont;
        meta = JsonUtility.FromJson<Meta>(Resources.Load<TextAsset>("sim/scene").text);
        ghostLeft = Resources.Load<TextAsset>("sim/ghost_left").text;
        ghostRight = Resources.Load<TextAsset>("sim/ghost_right").text;
        SimView.SurfaceFriction = meta.surface_friction;
        settings = Settings.Load();
        current = new Link.Params
        {
            @object = meta.@object, mass_scale = meta.mass_scale, max_force = meta.max_force, assist = meta.assist,
            rail = meta.rail
        };
        fingerOf = new int[Sim.Joints];
        string[] fingers = { "thumb", "index", "middle", "ring", "pinky" };
        for (int j = 0; j < Sim.Joints; j++) fingerOf[j] = Array.IndexOf(fingers, meta.joints[j].Split('_')[0]);
        tracked = new TrackedHand(ghostLeft, ghostRight);
        grabber = new GameObject("grabber").transform;

        deskRoot = new GameObject("desk root").transform;
        room = new Room(deskRoot, material, uiMaterial, new Vector2(meta.desk_half[0], meta.desk_half[1]),
            meta.desk_centre_y, 2 * meta.desk_half[2]);
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
        ray = new GameObject("ray", typeof(MeshFilter), typeof(MeshRenderer)).transform;
        ray.GetComponent<MeshFilter>().sharedMesh = Ui.RoundedRect(new Vector2(1, 1), 0, Ui.Accent);
        ray.GetComponent<MeshRenderer>().sharedMaterial = uiMaterial;
        ray.gameObject.SetActive(false);
        BuildMenu();
        BuildCard();
        BuildDock();
        BuildGhost();
        SetPassthrough(settings.passthrough);

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
        // One scene per prosthetic side and rail (export_model.py)
        var xml = Resources.Load<TextAsset>($"sim/scene_{(settings.prostheticLeft ? "left" : "right")}" +
            (current.rail == "gradient" ? "_gradient" : "")).text;
        sim.Load(xml, w =>
        {
            Native.bb_set_floor(w, height);
            Native.bb_set_params(w, p.mass_scale, p.max_force, p.assist ? 1 : 0);
            if (first.i >= 0) Native.bb_place_object(w, first.i, first.pos, new double[] { 1, 0, 0, 0 }, 0);
        });
        if (first.i >= 0) parked[first.i] = false;
    }

    // Menu, card, dock and ghost

    void BuildMenu()
    {
        menu = new Ui(uiMaterial);
        var play = menu.AddPage("Play");
        menu.Section(play, "Status");
        linkText = menu.Paragraph(play, "", 0.014f, false);
        simText = menu.Paragraph(play, "");
        readouts = menu.Readouts(play, "prosthetic grip", "free hand grip");
        menu.Section(play, "Haptics");
        hapticsText = menu.Paragraph(play, "", 0.014f, false);
        menu.Section(play, "Physics");
        massText = menu.Slider(play, "Object mass", 0.25f, 3, 0.25f, () => current.mass_scale,
            v => Change(p => p.mass_scale = v));
        forceText = menu.Slider(play, "Hand strength", 2, 40, 1, () => current.max_force, v => Change(p => p.max_force = v));
        menu.Toggle(play, "Grasp assist", () => current.assist, () => Change(p => p.assist = !p.assist));
        menu.Toggle(play, "Gradient rail", () => current.rail == "gradient",
            () => SetRail(current.rail == "gradient" ? "steps" : "gradient"));
        menu.Toggle(play, "Passthrough", () => settings.passthrough, () => SetPassthrough(!settings.passthrough));

        var objects = menu.AddPage("Objects");
        menu.Section(objects, "Spawn on the desk");
        menu.Buttons(objects, ("Can", () => Spawn("can")), ("Ball", () => Spawn("ball")), ("Block", () => Spawn("block")));
        menu.Buttons(objects, ("Cube", () => Spawn("small cube")), ("Pen", () => Spawn("pen")), ("Clear", ClearDesk));
        menu.Section(objects, "Desk");
        menu.Toggle(objects, "Move it with your hand", () => movingDesk, () => movingDesk = !movingDesk);
        heightText = menu.Slider(objects, "Height", 50, 110, 1, () => Mathf.Round(settings.deskHeight * 100),
            v => SetDeskHeight(v / 100));
        menu.Buttons(objects, ("Put it in front of me", DeskToMe));

        var rec = menu.AddPage("Record");
        menu.Section(rec, "Protocol F");
        menu.Paragraph(rec, "The laptop records as it does from the Record tab, with the setup of your last session. " +
                            "Each cue shows on the card over the desk.", 0.013f, true, 3);
        menu.Primary(rec, "Quick calibration", () => SendRecord("start", true));
        menu.Buttons(rec, ("Full session", () => SendRecord("start", false)));
        menu.Section(rec, "Session");
        recordText = menu.Paragraph(rec, "", 0.014f, false);
        recordError = menu.Paragraph(rec, "", 0.013f, false, 2);
        sessionButtons = menu.ButtonsWithLabels(rec, ("Pause", PauseOrContinue), ("Mark bad", () => SendRecord("bad")),
            ("Stop", () => SendRecord("stop")));
        menu.Section(rec, "Train");
        trainText = menu.Paragraph(rec, "", 0.013f, true, 2);
        trainError = menu.Paragraph(rec, "", 0.013f, false, 2);
        menu.Buttons(rec, ("Calibrate on it", () => SendTrain("calibrate")), ("Control with it", () => SendTrain("control")));

        var setup = menu.AddPage("Setup");
        menu.Section(setup, "Prosthetic arm");
        sideText = menu.Paragraph(setup, "", 0.014f, false);
        menu.Buttons(setup, ("Left arm", () => SetSide(true)), ("Right arm", () => SetSide(false)));
        menu.Section(setup, "Placing its hand");
        menu.Paragraph(setup, "Turn on passthrough to see your arm. With adjusting on, make a fist inside the virtual " +
                              "hand with your other hand and move it to where it should sit on the strapped controller.",
            0.013f, true, 4);
        menu.Toggle(setup, "Adjust by hand", () => adjusting, () => adjusting = !adjusting);
        menu.Toggle(setup, "Passthrough", () => settings.passthrough, () => SetPassthrough(!settings.passthrough));
        menu.Toggle(setup, "Show controllers", () => showControllers, () => showControllers = !showControllers);
        menu.Buttons(setup, ("Reset the hand", ResetOffset));
        menu.Section(setup, "Free hand");
        freeText = menu.Paragraph(setup, "", 0.013f, true, 2);
        menu.Show(play);
        menu.Close();
    }

    void BuildCard()
    {
        card = new Ui(uiMaterial, false);
        var page = card.AddPage("");
        cueMeta = card.Paragraph(page, "", 0.011f, true);
        cueTitle = card.Paragraph(page, "", 0.028f, false);
        cueBody = card.Paragraph(page, "", 0.014f, true, 3);
        cueProgress = card.Progress(page);
        cueNext = card.Paragraph(page, "", 0.012f, true);
        cueState = card.Paragraph(page, "", 0.014f, false, 2);
        card.Primary(page, "Continue", () => SendRecord("continue"));
        card.Buttons(page, ("Pause", () => SendRecord("pause")), ("Mark bad", () => SendRecord("bad")),
            ("Stop", () => SendRecord("stop")));
        card.Show(page);
        card.root.SetParent(deskRoot, false);
        card.root.localPosition = new Vector3(0, 0.42f, 0.5f);
        card.root.localRotation = Quaternion.Euler(-12, 0, 0);
        card.Close();
    }

    void BuildDock()
    {
        dock = new Ui(uiMaterial, false);
        var page = dock.AddPage("");
        dock.Buttons(page, ("Menu", ToggleMenu));
        dock.Show(page);
        dock.root.SetParent(deskRoot, false);
        dock.root.localScale = Vector3.one * 0.45f;
        PlaceDock();
        dock.root.gameObject.SetActive(true);
    }

    // At the desk's near corner on the free hand's side, tilted up to be read and touched
    void PlaceDock()
    {
        float x = (FreeIsRight ? 1 : -1) * (meta.desk_half[0] - 0.12f);
        dock.root.localPosition = new Vector3(x, 0.05f, -meta.desk_centre_y - meta.desk_half[1] + 0.05f);
        dock.root.localRotation = Quaternion.Euler(50, 0, 0);
    }

    void BuildGhost()
    {
        ghost?.Destroy();
        ghost = new Ghost(settings.prostheticLeft ? ghostLeft : ghostRight, material, SimView.Hex(0x6b6b75));
        ghost.Root.SetParent(deskRoot, false);
        // Upright beside the card on the prosthetic side, the back of the hand towards the user, like looking at
        // your own raised hand
        ghost.Root.localPosition = new Vector3((settings.prostheticLeft ? -1 : 1) * 0.33f, 0.3f, 0.5f);
        ghost.Root.localRotation = Quaternion.Euler(90, 0, 0);
        ghost.Root.localScale = Vector3.one * 1.2f;
        ghost.Root.gameObject.SetActive(false);
        reach?.Destroy();
        reach = new Ghost(settings.prostheticLeft ? ghostLeft : ghostRight, material, SimView.Hex(0x6b6b75));
        reach.Root.gameObject.SetActive(false);
    }

    void ToggleMenu()
    {
        if (menu.Visible) menu.Close();
        else menu.Open(head);
    }

    void SendRecord(string action, bool quick = false) =>
        link.Send(JsonUtility.ToJson(new RecordMessage { record = new RecordAction { action = action, quick = quick } }));

    void SendTrain(string action) =>
        link.Send(JsonUtility.ToJson(new TrainMessage { train = new TrainAction { action = action } }));

    void PauseOrContinue() => SendRecord(record != null && record.paused ? "continue" : "pause");

    void ResetOffset()
    {
        settings.prosthetic.pos = Settings.HeldPos;
        settings.prosthetic.rot = Vector3.zero;
        settings.Save();
    }

    void SetSide(bool left)
    {
        if (settings.prostheticLeft == left) return;
        settings.prostheticLeft = left;
        settings.Save();
        LoadScene();
        BuildGhost();
        PlaceDock();
    }

    void SetPassthrough(bool on)
    {
        settings.passthrough = on;
        settings.Save();
        if (passthroughCamera != null) passthroughCamera.enabled = on;
        // Passthrough shows through where the app draws nothing: a clear background with no alpha
        eyes.backgroundColor = on ? new Color(0, 0, 0, 0) : new Color(Ui.Bg.r, Ui.Bg.g, Ui.Bg.b, 1);
        room.SetVisible(!on);
        RenderSettings.fog = !on;
    }

    // A change made here goes to the sim and to the laptop, which shares it with its own page
    void Change(Action<Link.Params> edit)
    {
        edit(current);
        ApplyParams();
        link.Send(JsonUtility.ToJson(new ParamsMessage { @params = current }));
    }

    // Another rail is another model, as on the laptop: the scene starts over
    void SetRail(string rail)
    {
        Change(p => p.rail = rail);
        LoadScene();
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
        if (!string.IsNullOrEmpty(p.rail) && p.rail != current.rail)
        {
            current.rail = p.rail;
            LoadScene();
        }
        if (laptopObject != null && p.@object != laptopObject && Array.IndexOf(meta.objects, p.@object) >= 0)
        {
            ClearDesk();
            Spawn(p.@object);
        }
        laptopObject = p.@object;
        current.@object = p.@object;
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
        // Turns about where the grab started and moves with the hand, level: the height is set on the menu
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
        if (!d.isValid) return false;
        if (!eye && d.TryGetFeatureValue(CommonUsages.isTracked, out bool on) && !on) return false;
        return d.TryGetFeatureValue(eye ? CommonUsages.centerEyePosition : CommonUsages.devicePosition, out p)
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

    // The hand as it holds a controller (fingers forward, palm towards the controller's middle, thumb up)
    static Quaternion Held(bool leftHand) => Quaternion.LookRotation(Vector3.back, leftHand ? Vector3.right : Vector3.left);

    // The wrist pose for a hand on its controller, turned and moved by the setup's offset
    static void HandPose(Vector3 cp, Quaternion cq, bool leftHand, Settings.Offset offset, out Vector3 p, out Quaternion q)
    {
        q = cq * Quaternion.Euler(offset.rot) * Held(leftHand);
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

        tracked.Update(FreeIsRight);
        bool hasP = Pose(ProstheticNode, out var pp, out var pq);
        Vector3 fp = default;
        Quaternion fq = Quaternion.identity;
        bool hasF = !tracked.Tracked && Pose(FreeNode, out fp, out fq);
        controllerModels[0].gameObject.SetActive(hasP && showControllers);
        controllerModels[1].gameObject.SetActive(hasF && showControllers);
        if (hasP) controllerModels[0].SetPositionAndRotation(pp, pq);
        if (hasF) controllerModels[1].SetPositionAndRotation(fp, fq);

        // The free hand: the tracked hand if it's seen, the controller otherwise
        float grip = 0, trigger = 0;
        if (tracked.Tracked)
        {
            grabber.SetPositionAndRotation(tracked.WristPos, tracked.WristRot);
            PlaceHand(Sim.Free, tracked.WristPos, tracked.WristRot);
            grip = tracked.Closure;
        }
        else if (hasF)
        {
            grabber.SetPositionAndRotation(fp, fq);
            HandPose(fp, fq, !settings.prostheticLeft, settings.free, out var p, out var q);
            PlaceHand(Sim.Free, p, q);
            grip = Axis(FreeNode, CommonUsages.grip);
            trigger = Axis(FreeNode, CommonUsages.trigger);
        }

        // The menu, card and dock, by fingertip or by controller ray
        bool menuButton = hasF && Button(FreeNode, CommonUsages.secondaryButton);
        if (menuButton && !prevMenuButton) ToggleMenu();
        prevMenuButton = menuButton;
        // With the tracked hand: palm turned to the face and the thumb on the middle fingertip. The index pinch there is
        // the headset's own menu
        bool menuGesture = tracked.Tracked && tracked.MiddlePinching
                           && Vector3.Dot(tracked.PalmNormal, (head.position - tracked.WristPos).normalized) > 0.5f;
        if (menuGesture && !prevMenuGesture) ToggleMenu();
        prevMenuGesture = menuGesture;
        Vector3? tip = tracked.Tracked ? tracked.IndexTip : (Vector3?)null;
        Ray? pointer = hasF ? new Ray(fp, fq * Vector3.forward) : (Ray?)null;
        bool grabHandle = tracked.Tracked ? tracked.Pinching : grip > GrabOn;
        float length = 0;
        bool clicked = false;
        foreach (var panel in new[] { menu, card, dock })
        {
            float l = panel.Interact(tip, pointer, trigger > 0.7f, grabHandle, grabber, out bool c);
            if (l > 0 && length == 0) length = l;
            clicked |= c;
        }
        if (clicked && hasF) Buzz(FreeNode, 0.25f, 0.03f);
        bool showRay = hasF && (menu.Visible || length > 0);
        ray.gameObject.SetActive(showRay);
        if (showRay)
        {
            float shown = length > 0 ? length : 0.6f;
            ray.SetPositionAndRotation(fp + fq * Vector3.forward * (shown / 2), fq * Quaternion.Euler(90, 0, 0));
            ray.localScale = new Vector3(0.0015f, shown, 1);
        }
        bool busy = length > 0 || menu.Grabbed || card.Grabbed;

        // Placing the prosthetic hand: a fist inside it carries it, its offset from the controller follows
        bool fist = grip > GrabOn;
        if (adjusting && hasP && (tracked.Tracked || hasF)) AdjustProsthetic(pp, pq, fist && !busy);
        else adjustDragging = false;
        if (hasP)
        {
            HandPose(pp, pq, settings.prostheticLeft, settings.prosthetic, out var p, out var q);
            PlaceHand(Sim.Prosthetic, p, q);
            reach.Root.SetPositionAndRotation(p, q);
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
        record = frame?.record;

        if (movingDesk && (tracked.Tracked || hasF)) DragDesk(grabber.position, grabber.rotation, fist && !busy);
        FreeHandInput(grip, trigger, busy || movingDesk || adjustDragging);

        bool heldBack = reach.Root.gameObject.activeSelf ? readout[17] > ReachHideM : readout[17] > ReachShowM;
        reach.Root.gameObject.SetActive(hasP && heldBack);
        if (hasP && heldBack) reach.Pose(frame != null ? frame.angles : meta.rest);

        if (view.Update(sim, readout, parked))
        {
            int held = (int)readout[9];
            if (held >= 0 && heldBefore < 0 && hasF) Buzz(FreeNode, 0.4f, 0.04f);
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
        ShowRecording();
        ShowLabels(frame);
    }

    void AdjustProsthetic(Vector3 cp, Quaternion cq, bool fist)
    {
        HandPose(cp, cq, settings.prostheticLeft, settings.prosthetic, out var handPos, out var handRot);
        if (!adjustDragging)
        {
            if (!fist || Vector3.Distance(grabber.position, handPos) > AdjustReach) return;
            adjustDragging = true;
            adjustOffsetPos = Quaternion.Inverse(grabber.rotation) * (handPos - grabber.position);
            adjustOffsetRot = Quaternion.Inverse(grabber.rotation) * handRot;
            return;
        }
        if (!fist)
        {
            adjustDragging = false;
            settings.Save();
            return;
        }
        var newPos = grabber.position + grabber.rotation * adjustOffsetPos;
        var newRot = grabber.rotation * adjustOffsetRot;
        settings.prosthetic.pos = Quaternion.Inverse(cq) * (newPos - cp);
        settings.prosthetic.rot = (Quaternion.Inverse(cq) * newRot * Quaternion.Inverse(Held(settings.prostheticLeft))).eulerAngles;
    }

    // The free hand's fingers: the tracked hand's own angles, or from the controller (grip closes the middle, ring and
    // little fingers, the trigger the index, the thumb comes down on the buttons). Closing it holds what it touches,
    // except while it works the menu, moves the desk or places the other hand
    void FreeHandInput(float grip, float trigger, bool busy)
    {
        if (tracked.Tracked)
        {
            Array.Copy(tracked.Angles, freeAngles, Sim.Joints);
        }
        else
        {
            var node = FreeNode;
            bool thumbDown = Button(node, CommonUsages.primaryTouch) || Button(node, CommonUsages.secondaryTouch)
                             || Button(node, CommonUsages.primary2DAxisTouch);
            for (int j = 0; j < Sim.Joints; j++)
            {
                int f = fingerOf[j];
                float amount = f == 1 ? trigger : f == 0 ? (thumbDown ? Mathf.Max(grip, 0.5f) : 0.2f * grip) : grip;
                freeAngles[j] = Mathf.Lerp(meta.rest[j], meta.fist[j], amount);
            }
        }
        if (busy) grip = trigger = 0;
        grabbing = grabbing ? grip > GrabOff : grip > GrabOn;
        sim.SetInput(Sim.Free, freeAngles, Mathf.Max(grip, trigger), false, grabbing);
    }

    // The cue card and ghost hand while the laptop records
    void ShowRecording()
    {
        bool running = record != null && record.running;
        if (running != card.Visible)
        {
            if (running) card.root.gameObject.SetActive(true);
            else card.Close();
        }
        ghost.Root.gameObject.SetActive(running && record.has_target);
        if (!running) return;
        if (record.index != cueIndex)
        {
            cueIndex = record.index;
            cueStartedAt = Time.time - record.elapsed;
        }
        var lines = (record.text ?? "").Split('\n');
        cueMeta.text = $"CUE {record.index + 1} / {record.count}" + (record.marked_bad ? "   ·   <color=#ff3b3b>MARKED BAD</color>" : "");
        cueTitle.text = lines[0];
        cueBody.text = lines.Length > 1 ? string.Join(" ", lines, 1, lines.Length - 1) : "";
        cueProgress(record.seconds > 0 ? record.elapsed / record.seconds : 0);
        cueNext.text = string.IsNullOrEmpty(record.next) ? "" : "Next: " + record.next;
        cueState.text = record.hum ? "<color=#ffc233>Hum on the band: the cue will be redone. Check the pads, then Continue.</color>"
            : record.paused ? "<color=#ffc233>Paused. Continue when ready.</color>"
            : record.seconds <= 0 ? "Continue when ready." : "";
        if (record.has_target && record.target != null && record.target.Length == Sim.Joints)
        {
            float t = Mathf.Clamp01((Time.time - cueStartedAt) / FollowInS);
            for (int j = 0; j < Sim.Joints; j++) ghostAngles[j] = Mathf.Lerp(meta.rest[j], record.target[j], t);
            ghost.Pose(ghostAngles);
        }
    }

    void ShowLabels(Link.Frame frame)
    {
        bool driven = frame != null && frame.driven;
        linkText.text = !link.Connected ? "Laptop not connected, the hand rests"
            : driven ? (frame.locked ? "Control running, locked" : "Control running")
            : "Laptop connected, no control running";
        bool recording = record != null && record.running;
        Ui.SetChip(menu, recording ? "RECORDING" : link.Connected ? "LAPTOP" : "OFFLINE",
            recording ? Ui.Bad : link.Connected ? Ui.Good : Ui.Muted);
        simText.text = sim.Error ?? $"Sim {sim.Speed:0.0}x real time";
        hapticsText.text = HapticsText(frame?.haptics);
        readouts[0].text = $"{readout[0]:0.0} N";
        readouts[1].text = $"{readout[7]:0.0} N";
        massText.text = $"x{current.mass_scale:0.00}";
        forceText.text = $"{current.max_force:0} N";
        heightText.text = $"{settings.deskHeight * 100:0} cm";
        sideText.text = settings.prostheticLeft ? "Left: the controller strapped to the left forearm"
            : "Right: the controller strapped to the right forearm";
        freeText.text = tracked.Tracked ? "Tracked from your real hand, nothing to set."
            : "Not seen. Hold it in view, or hold a controller in it.";
        recordText.text = record == null ? "Laptop not connected"
            : recording ? $"Cue {record.index + 1} of {record.count}" + (record.paused ? ", paused" : "")
            : "Not recording";
        recordError.text = record != null && !string.IsNullOrEmpty(record.error) ? $"<color=#ff3b3b>{record.error}</color>" : "";
        sessionButtons[0].text = record != null && record.paused ? "Continue" : "Pause";
        var train = frame?.train;
        trainText.text = train == null ? "Laptop not connected"
            : !string.IsNullOrEmpty(train.status) ? train.status
            : !string.IsNullOrEmpty(train.session) ? "Fits the sign network to where the band sits now, on the session just recorded."
            : "Record a quick calibration, then calibrate the sign network on it here.";
        trainError.text = train != null && !string.IsNullOrEmpty(train.error) ? $"<color=#ff3b3b>{train.error}</color>" : "";
    }

    static string HapticsText(Link.Haptics h)
    {
        if (h == null || !h.on) return "Motors off. Start them from Grip control on the laptop.";
        if (h.mode != "surface") return $"Following effort, strongest output {h.duty:0}%";
        if (string.IsNullOrEmpty(h.surface)) return h.test ? "Test touch, not touching yet" : "Not touching anything";
        string friction = h.friction >= 0 ? $", friction {h.friction:0.00}" : "";
        // No duty here: it pulses on and off with the surface and would flicker
        return $"{h.force:0.0} N on {h.surface}{friction}, {h.hz:0.#} Hz pulses" + (h.test ? ", test touch" : "");
    }

    void SendReadout()
    {
        var r = new Readout
        {
            grip_force = (float)readout[0], finger_force = new float[5], palm_force = (float)readout[6],
            free_grip_force = (float)readout[7], speed = (float)sim.Speed, blowups = (int)readout[13],
            touch_force = (float)readout[14], touch_friction = (float)readout[15], touch_surface = (int)readout[16]
        };
        for (int i = 0; i < 5; i++) r.finger_force[i] = (float)readout[1 + i];
        link.Readout = JsonUtility.ToJson(new ReadoutMessage { readout = r });
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
                    type = GeomTypes[src.type],
                    role = src.role == 1 ? "object" : src.role == 2 ? "hand" : src.role == 3 ? "rail" : "static",
                    surface = src.surface >= 0 && src.surface < SurfaceNames.Length ? SurfaceNames[src.surface] : "",
                    friction = src.friction,
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
        viewOut.menu = menu.Visible;
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
