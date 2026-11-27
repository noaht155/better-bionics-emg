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
    // Quit takes a second press within this long, so a stray touch doesn't close the app
    const float QuitConfirmS = 3;
    // Off the head this long, the app closes itself
    const int QuitAfterOffMs = 10 * 60 * 1000;
    System.Threading.Timer quitTimer;
    bool linkStopped;
    // The Help page opens this long after the intro on the first start
    const float HelpAfterS = 0.5f;

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
        public double echo;
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
        // The panels in view and the pointing ray (start and end), for the laptop's VR tab, in sim coordinates
        public Ui.PanelOut[] panels;
        public float[] ray;
        public string[] menu_pages;
        // Both real hands' joint angles while the headset tracks them (HandAngles, the camera's definitions), for
        // the laptop to keep beside a session: the free hand and the one on the prosthetic side
        public bool free_tracked, other_tracked;
        public float[] free_angles, other_angles;
    }

    [Serializable] class ViewMessage { public View view; }

    [Serializable]
    class RecordAction
    {
        public string action;
        public bool quick;
    }

    [Serializable] class RecordMessage { public RecordAction record; }

    [Serializable] class TrainAction { public string action, name; }

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
    Ui menu, card, dock, controls, board;
    // The board behind the desk while control runs, the Control tab's model output and readout (Noah, 2026-11-24):
    // a bar per class on the left, the grip line and the numbers on the right. Rebuilt when the classes change
    TextMeshPro boardTitle, boardSub;
    TextMeshPro[] boardRows, boardNumbers;
    Action<float>[] boardBars;
    string boardClasses;
    Link.Frame controlFrame;
    readonly Sounds sounds = new Sounds();
    Intro intro;
    float introDoneAt = float.PositiveInfinity;
    // The bare hand's ray, smoothed: hand tracking shakes a few mm, which a 1 m ray turns into cm
    Vector3 rayOrigin, rayDir;
    bool rayWarm;
    float rayShownUntil;
    bool triggerDown;
    // ghost: the cue's sign in front. The prosthetic hand itself shows the fingers. Nothing sits on either hand, an
    // overlay there confused (Noah, 2026-11-24)
    Ghost ghost, reach;
    // The ghost's forearm grows in and out for the wrist cues instead of popping (Noah, 2026-11-24)
    float armShown;
    const float ArmFadeS = 0.3f;
    Ui.Page help;
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
    readonly Ui.PanelOut[] panelsOut = { new Ui.PanelOut(), new Ui.PanelOut(), new Ui.PanelOut(), new Ui.PanelOut() };
    Tour tour;
    // What the tour's steps wait for
    bool cubeSpawned, menuMoved, menuWasGrabbed;
    TrackedHand otherHand;
    readonly float[] rayOut = new float[6];
    bool rayOn;
    bool grabbing, prevMenuButton, movingDesk, deskDragging, adjusting, adjustDragging, showControllers, wasConnected;
    Vector3 dragStart, deskStart, adjustOffsetPos;
    Quaternion adjustOffsetRot;
    float adjustDistance;
    bool adjustHover, adjustArmed;
    float dragYaw, deskYawStart;
    readonly List<XRInputSubsystem> inputs = new List<XRInputSubsystem>();
    bool floorSet, recentered;

    TextMeshPro linkText, simText, hapticsText, trainText, trainError, calText, massText, forceText, heightText, sideText, freeText, recordText, recordError;
    TextMeshPro[] readouts, sessionButtons, quitButton;
    float quitArmedAt = -QuitConfirmS;
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
        otherHand = new TrackedHand(ghostLeft, ghostRight);
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
        BuildControls();
        BuildTour();
        BuildGhost();
        SetPassthrough(settings.passthrough);
        // The room, desk and hands appear once the intro is done
        intro = new Intro(uiMaterial);
        deskRoot.gameObject.SetActive(false);
        room.SetVisible(false);

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
        menu = new Ui(uiMaterial, true, 0.28f);
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
        menu.Section(play, "App");
        quitButton = menu.ButtonsWithLabels(play, ("Quit", Quit));

        var objects = menu.AddPage("Objects");
        menu.Section(objects, "Spawn on the desk");
        menu.Buttons(objects, ("Can", () => Spawn("can")), ("Ball", () => Spawn("ball")), ("Block", () => Spawn("block")));
        menu.Buttons(objects, ("Cube", () => Spawn("small cube")), ("Pen", () => Spawn("pen")), ("Clear", ClearDesk));
        menu.Section(objects, "Desk");
        menu.Toggle(objects, "Move it with your hand", () => movingDesk, () => movingDesk = !movingDesk);
        heightText = menu.Slider(objects, "Height", 40, 130, 1, () => Mathf.Round(settings.deskHeight * 100),
            v => SetDeskHeight(v / 100));
        menu.Buttons(objects, ("Put it in front of me", DeskToMe), ("Set height by hand", StartHandHeight));

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
        menu.Buttons(rec, ("Calibrate on it", () => SendTrain("calibrate")));
        calText = menu.Paragraph(rec, "", 0.013f, true, 2);
        menu.Buttons(rec, ("Older", () => PickCalibration(1)), ("Newer", () => PickCalibration(-1)),
            ("Control with it", () => SendTrain("control", calPicked)));

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

        help = menu.AddPage("Help");
        menu.Primary(help, "Start the tour", () =>
        {
            menu.Close();
            StartTour();
        });
        menu.Section(help, "This menu");
        menu.Paragraph(help, "Open and close it with B or Y on the free hand's controller, the Menu button at the " +
                             "desk's corner, or with your bare hand: palm towards your face, thumb on the middle fingertip.",
            0.013f, true, 4);
        menu.Paragraph(help, "Touch it with a fingertip, or point the controller and pull the trigger. Pinch or grip " +
                             "the bar under it to move it.", 0.013f, true, 3);
        menu.Section(help, "Hands");
        menu.Paragraph(help, "The prosthetic hand sits on the controller strapped to that arm and its fingers follow " +
                           "the EMG control on the laptop. A grey hand shows where your arm is when the desk or an " +
                           "object holds the hand back.", 0.013f, true, 4);
        menu.Paragraph(help, "The free hand is your real hand, or a controller in it: grip and trigger close it to pick " +
                           "things up.", 0.013f, true, 3);
        menu.Toggle(help, "Controls card at the desk", () => settings.controlsCard, () => ShowControls(!settings.controlsCard));
        menu.Section(help, "Leaving");
        menu.Paragraph(help, "Quit is at the bottom of Play. Taking the headset off pauses the app, and after 10 " +
                             "minutes off it closes.", 0.013f, true, 3);
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
        card.Close();
        BuildBoard(new string[0]);
    }

    void BuildBoard(string[] classes)
    {
        bool placed = board != null;
        if (placed) Destroy(board.root.gameObject);
        boardClasses = string.Join(",", classes);
        // Two columns: the second section always starts the right one
        board = new Ui(uiMaterial, false, 0.001f);
        var page = board.AddPage("");
        board.Section(page, "Model output");
        boardRows = new TextMeshPro[classes.Length];
        boardBars = new Action<float>[classes.Length];
        for (int i = 0; i < classes.Length; i++)
        {
            boardRows[i] = board.Paragraph(page, "", 0.014f, true);
            boardBars[i] = board.Progress(page);
        }
        board.Section(page, "Control");
        boardTitle = board.Paragraph(page, "", 0.03f, false);
        boardSub = board.Paragraph(page, "", 0.012f, true);
        var top = board.Readouts(page, "effort", "closure", "starts");
        var bottom = board.Readouts(page, "releases", "open hand picked", "distance");
        boardNumbers = new[] { top[0], top[1], top[2], bottom[0], bottom[1], bottom[2] };
        board.Show(page);
        board.root.SetParent(deskRoot, false);
        if (placed) PlaceBoard();
        board.Close();
    }

    void PlaceBoard()
    {
        board.root.SetLocalPositionAndRotation(card.root.localPosition, card.root.localRotation);
        board.root.localScale = card.root.localScale;
    }

    void BuildDock()
    {
        dock = new Ui(uiMaterial, false);
        var page = dock.AddPage("");
        dock.Buttons(page, ("Menu", ToggleMenu));
        dock.Paragraph(page, "or B / Y", 0.012f);
        dock.Show(page);
        dock.root.SetParent(deskRoot, false);
        dock.root.localScale = Vector3.one * 0.45f;
        PlaceDock();
        dock.root.gameObject.SetActive(true);
    }

    // The controls at a glance, on a card by the Menu button
    void BuildControls()
    {
        controls = new Ui(uiMaterial, false);
        var page = controls.AddPage("");
        controls.Section(page, "Controls");
        controls.Paragraph(page, "<b>Menu</b>  B or Y. Bare hand: palm to your face, thumb on the middle fingertip.",
            0.014f, true, 2);
        controls.Paragraph(page, "<b>Select</b>  touch with a fingertip, or point and pull the trigger or pinch.",
            0.014f, true, 2);
        controls.Paragraph(page, "<b>Grab</b>  grip, or close your hand. Move a panel by the bar under it.", 0.014f, true, 2);
        controls.Show(page);
        controls.root.SetParent(deskRoot, false);
        controls.root.localScale = Vector3.one * 0.5f;
        PlaceControls();
        controls.root.gameObject.SetActive(settings.controlsCard);
    }

    void PlaceControls()
    {
        // Beside the Menu button on the free hand's side, just past the desk's near edge towards the user and tilted
        // up to the face (Noah: closer)
        float x = (FreeIsRight ? 1 : -1) * (meta.desk_half[0] - 0.12f - 0.09f - Ui.Width * 0.5f / 2);
        controls.root.localPosition = new Vector3(x, 0.1f, -meta.desk_centre_y - meta.desk_half[1] - 0.1f);
        controls.root.localRotation = Quaternion.Euler(35, 0, 0);
    }

    void ShowControls(bool on)
    {
        settings.controlsCard = on;
        settings.Save();
        controls.root.gameObject.SetActive(on);
    }

    // At the desk's near corner on the free hand's side, tilted up to be read and touched
    void PlaceDock()
    {
        // The cue card as a big board behind the desk's far edge, clear of the hands (Noah, 2026-11-24: over the desk
        // it covered the hand), with the ghost beside it
        card.root.localPosition = new Vector3(0, 0.5f, -meta.desk_centre_y + meta.desk_half[1] + 0.12f);
        card.root.localRotation = Quaternion.Euler(-5, 0, 0);
        card.root.localScale = Vector3.one * 1.5f;
        PlaceBoard();
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
        ghost.Root.localPosition = new Vector3((settings.prostheticLeft ? -1 : 1) * 0.45f, 0.3f, 0.5f);
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

    // Closed fully, as the Quest's own Quit does: paused, the sim and link threads keep running and drain the battery
    void Quit()
    {
        if (Time.time - quitArmedAt > QuitConfirmS)
        {
            quitArmedAt = Time.time;
            return;
        }
        settings.Save();
        Application.Quit();
    }

    void SendTrain(string action, string name = "") =>
        link.Send(JsonUtility.ToJson(new TrainMessage { train = new TrainAction { action = action, name = name ?? "" } }));

    // One of the laptop's newest calibrations (app.headset_calibrations, newest first) for Control with it, so an
    // older one can be used too (Noah, 2026-11-24). Stays on the one picked while new ones come in
    string calPicked;
    string[] calibrations = new string[0];

    void PickCalibration(int step)
    {
        if (calibrations.Length == 0) return;
        int i = Mathf.Clamp(Array.IndexOf(calibrations, calPicked) + step, 0, calibrations.Length - 1);
        calPicked = calibrations[i];
    }

    static string CalibrationText(string name)
    {
        int at = name.LastIndexOf("_cal_", StringComparison.Ordinal);
        var when = at < 0 ? name : name.Substring(at + 5).Replace(".pt", "");
        // 2026-11-24_141429 -> 2026-11-24 14:14
        if (when.Length >= 15 && when[10] == '_') when = when.Substring(0, 10) + " " + when.Substring(11, 2) + ":" + when.Substring(13, 2);
        return "Calibrated " + when;
    }

    Ray SteadyRay(Vector3 origin, Vector3 dir)
    {
        if (!rayWarm) (rayOrigin, rayDir, rayWarm) = (origin, dir, true);
        // Steadier the slower it turns, quick when it swings, like a one euro filter
        float turn = Vector3.Angle(rayDir, dir) / Mathf.Max(Time.deltaTime, 1e-4f);
        float tau = Mathf.Lerp(0.15f, 0.03f, Mathf.InverseLerp(20, 200, turn));
        float k = 1 - Mathf.Exp(-Time.deltaTime / tau);
        rayOrigin = Vector3.Lerp(rayOrigin, origin, k);
        rayDir = Vector3.Slerp(rayDir, dir, k).normalized;
        return new Ray(rayOrigin, rayDir);
    }

    // Whether the touch that started the current hold paused the recording, so the Pause it confirms doesn't
    // continue it again
    bool pausedByTouch;

    // A press through a ray is held to confirm (Noah, 2026-11-24: pinches set things off by accident); during a
    // recording the start of the hold pauses it, so a stray one costs a pause, not a cue
    void HoldStarted()
    {
        sounds.Play(Sounds.Kind.Tick, head.position);
        if (record != null && record.running) PauseForTouch();
    }

    void PauseForTouch()
    {
        pausedByTouch = record != null && !record.paused;
        if (pausedByTouch) SendRecord("pause");
    }

    void PauseOrContinue()
    {
        if (pausedByTouch)
        {
            pausedByTouch = false;
            return;
        }
        SendRecord(record != null && record.paused ? "continue" : "pause");
    }

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
        PlaceControls();
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
                else if (c.kind == "point") PointAt(c.arg);
                else if (c.kind == "menu") OpenMenuAt(c.arg);
                else if (c.kind == "tour") StartTour();
            }
            newest = Math.Max(newest, c.id);
        }
        lastCommand = Math.Max(newest, 0);
    }

    // The laptop points at a control to show the user where to press: "panel	page	name"
    void PointAt(string arg)
    {
        var parts = (arg ?? "").Split('	');
        if (parts.Length != 3) return;
        var panel = parts[0] == "menu" ? menu : parts[0] == "card" ? card : parts[0] == "dock" ? dock
            : parts[0] == "tour" ? tour.card : null;
        if (panel != null && panel.Pulse(parts[1], parts[2])) sounds.Play(Sounds.Kind.Tick, panel.root.position);
    }

    // The laptop opens the menu on a page, or closes it with an empty one
    void OpenMenuAt(string page)
    {
        if (string.IsNullOrEmpty(page))
        {
            menu.Close();
            return;
        }
        if (!menu.ShowPage(page)) return;
        if (!menu.Visible) menu.Open(head);
        sounds.Play(Sounds.Kind.Tick, menu.root.position);
    }

    void StartTour()
    {
        cubeSpawned = menuMoved = false;
        tour.Start();
    }

    void BuildTour()
    {
        var steps = new[]
        {
            new Tour.Step
            {
                text = "Open the menu: press B or Y on the controller in your free hand, or turn your palm to your " +
                       "face and touch your thumb to your middle fingertip.",
                done = () => menu.Visible, point = () => dock.Pulse("", "Menu")
            },
            new Tour.Step
            {
                text = "Pick the Objects tab at the top of the menu: touch it, or point at it and pull the trigger or " +
                       "pinch.",
                done = () => menu.Visible && menu.PageName == "Objects", point = () => menu.Pulse(menu.PageName, "Objects")
            },
            new Tour.Step
            {
                text = "Put a cube on the desk: hold Cube until the bar under it fills.",
                done = () => cubeSpawned, point = () => menu.Pulse("Objects", "Cube")
            },
            new Tour.Step
            {
                text = "Pick the cube up with your free hand: grip the controller, or close your real hand round it.",
                done = () => readout[9] >= 0
            },
            new Tour.Step
            {
                text = "Move the menu: grab the bar under it with a fist or a pinch, drag it and let go.",
                done = () => menuMoved, point = () => menu.Pulse(menu.PageName, "Move bar")
            },
            new Tour.Step
            {
                text = "Close the menu with the × in its corner.",
                done = () => !menu.Visible, point = () => menu.Pulse(menu.PageName, "Close")
            },
        };
        // Where the cue card goes, behind the desk; the card only shows while recording, which ends the tour
        tour = new Tour(uiMaterial, steps, ui =>
        {
            ui.root.SetParent(deskRoot, false);
            ui.root.SetLocalPositionAndRotation(card.root.localPosition, card.root.localRotation);
            ui.root.localScale = card.root.localScale;
        });
    }

    void ToSimFloats(Vector3 world, float[] to, int at)
    {
        SimView.ToSim(deskRoot.InverseTransformPoint(world), headSim);
        for (int k = 0; k < 3; k++) to[at + k] = (float)headSim[k];
    }

    void ToSimVector(Vector3 world, float[] to)
    {
        SimView.ToSim(deskRoot.InverseTransformVector(world), headSim);
        for (int k = 0; k < 3; k++) to[k] = (float)headSim[k];
    }

    Ui.PanelOut[] PanelsInView()
    {
        var shown = new List<Ui.PanelOut>();
        var all = new[] { (menu, "menu"), (card, "card"), (dock, "dock"), (tour.card, "tour") };
        for (int i = 0; i < all.Length; i++)
        {
            var (ui, name) = all[i];
            if (!ui.Visible) continue;
            var o = panelsOut[i];
            ui.Describe(o, name);
            ToSimFloats(ui.Body.position, o.pos, 0);
            ToSimVector(ui.Body.TransformVector(Vector3.right), o.right);
            ToSimVector(ui.Body.TransformVector(Vector3.up), o.up);
            shown.Add(o);
        }
        return shown.ToArray();
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
        if (kind == "small cube") cubeSpawned = true;
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
        settings.Save();
        PlaceDeskFromSettings();
    }

    bool handHeight, handHeightArmed;

    void StartHandHeight()
    {
        handHeight = true;
        handHeightArmed = false;
    }

    void SetDeskHeight(float height)
    {
        settings.deskHeight = Mathf.Clamp(height, 0.4f, 1.3f);
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
        // Once, on the first start: the tour, once the intro is done
        if (!settings.helpSeen && Time.time > introDoneAt + HelpAfterS)
        {
            StartTour();
            settings.helpSeen = true;
            settings.Save();
        }
        if (!floorSet)
        {
            SubsystemManager.GetSubsystems(inputs);
            foreach (var s in inputs) floorSet |= s.TrySetTrackingOriginMode(TrackingOriginModeFlags.Floor);
            // Holding the Meta button recenters the tracking space: the desk comes along
            if (floorSet) foreach (var s in inputs) s.trackingOriginUpdated += _ => recentered = true;
        }
        TrackHead();
        if (!intro.Done)
        {
            intro.Update(head, sim.Geoms != null, link.Connected);
            if (!intro.Done) return;
            // In front of the user at every start: the saved spot is in tracking space, which the headset sets up
            // anew each time, so it can be behind or beside them
            DeskToMe();
            deskRoot.gameObject.SetActive(true);
            room.SetVisible(!settings.passthrough);
            introDoneAt = Time.time;
        }
        if (recentered)
        {
            recentered = false;
            DeskToMe();
        }

        tracked.Update(FreeIsRight);
        otherHand.Update(!FreeIsRight);
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
        // Pointing: the controller's ray, or with the bare hand a ray from about the shoulder through the pinch, as
        // the Quest's own menus do, selected by pinching
        Ray? pointer = null;
        // Pulled past 0.7, let go under 0.4, so a trigger held near one level doesn't flicker
        bool down = triggerDown = triggerDown ? trigger > 0.4f : trigger > 0.7f;
        if (hasF) pointer = new Ray(fp, fq * Vector3.forward);
        else if (tracked.Tracked)
        {
            var side = Vector3.ProjectOnPlane(head.right, Vector3.up).normalized * (FreeIsRight ? 1 : -1);
            var shoulder = head.position + Vector3.down * 0.15f + side * 0.15f;
            pointer = SteadyRay(tracked.IndexKnuckle, (tracked.IndexKnuckle - shoulder).normalized);
            down = tracked.Pinching;
        }
        if (hasF || !tracked.Tracked) rayWarm = false;
        bool grabHandle = tracked.Tracked ? tracked.Pinching : grip > GrabOn;
        float length = 0;
        bool clicked = false, hovered = false;
        foreach (var panel in new[] { menu, card, dock, tour.card })
        {
            float l = panel.Interact(tip, pointer, down, grabHandle, grabber, out bool c, true, false, HoldStarted);
            if (l > 0 && length == 0) length = l;
            clicked |= c;
            hovered |= panel.HoverChanged;
        }
        if (clicked)
        {
            if (hasF) Buzz(FreeNode, 0.25f, 0.03f);
            sounds.Play(Sounds.Kind.Click, head.position);
        }
        else if (hovered)
        {
            // Onto a new control: a faint tick and the lightest buzz, so moving along the menu can be felt
            if (hasF) Buzz(FreeNode, 0.08f, 0.01f);
            sounds.Play(Sounds.Kind.Hover, head.position);
        }
        // The bare hand's ray only shows while it points at a panel, so it isn't always in view
        // Kept a moment after it leaves a panel, so a shaky hand at the edge doesn't flicker it
        if (length > 0.0001f) rayShownUntil = Time.time + 0.3f;
        bool showRay = pointer.HasValue && (hasF && menu.Visible || Time.time < rayShownUntil);
        ray.gameObject.SetActive(showRay);
        if (showRay)
        {
            var r = pointer.Value;
            float shown = length > 0.0001f ? length : 0.6f;
            ray.SetPositionAndRotation(r.origin + r.direction * (shown / 2), Quaternion.LookRotation(r.direction) * Quaternion.Euler(90, 0, 0));
            ray.localScale = new Vector3(0.0015f, shown, 1);
            ToSimFloats(r.origin, rayOut, 0);
            ToSimFloats(r.origin + r.direction * shown, rayOut, 3);
        }
        rayOn = showRay;
        bool busy = length > 0 || menu.Grabbed || card.Grabbed;
        // Height by hand: the hand held out flat where the desk should be, set with a pinch or trigger pull away
        // from the panels. Waits for the press that started it to end first
        if (handHeight)
        {
            bool set = tracked.Tracked ? tracked.Pinching : hasF && trigger > 0.7f;
            if (!set) handHeightArmed = true;
            else if (handHeightArmed && length == 0)
            {
                float y = tracked.Tracked ? tracked.WristPos.y : fp.y;
                SetDeskHeight(y - 0.03f);
                handHeight = false;
                sounds.Play(Sounds.Kind.Click, head.position);
            }
        }

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
            // While a recording shows a sign, the prosthetic hand makes it too, so the user sees it on that hand
            if (Training) sim.SetInput(Sim.Prosthetic, ghostAngles, 0, false, false);
            else sim.SetInput(Sim.Prosthetic, frame.angles, frame.effort, frame.locked, false);
        }
        else
        {
            sim.SetInput(Sim.Prosthetic, meta.rest, 0, false, false);
        }
        record = frame?.record;
        controlFrame = frame;

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
        if (menu.Grabbed) menuWasGrabbed = true;
        else if (menuWasGrabbed) (menuWasGrabbed, menuMoved) = (false, true);
        if (tour.Update()) sounds.Play(Sounds.Kind.Click, head.position);
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

        void ShowControl(Link.Frame f)
    {
        var c = f?.control;
        bool on = c != null && c.classes != null && c.classes.Length > 0;
        if (on && string.Join(",", c.classes) != boardClasses) BuildBoard(c.classes);
        if (on != board.Visible)
        {
            if (on) board.root.gameObject.SetActive(true);
            else board.Close();
        }
        if (!on) return;
        for (int i = 0; i < boardRows.Length; i++)
        {
            float p = c.probs != null && i < c.probs.Length ? c.probs[i] : 0;
            // Over the threshold in white, as the Control tab's line marks it
            boardRows[i].text = $"{c.classes[i]}   {p * 100:0} %";
            boardRows[i].color = p >= c.threshold ? Ui.Text : Ui.Muted;
            boardBars[i](p);
        }
        boardTitle.text = c.title;
        boardSub.text = c.sub;
        boardNumbers[0].text = $"{c.effort * 100:0} %";
        boardNumbers[1].text = $"{c.closure * 100:0} %";
        boardNumbers[2].text = c.starts.ToString();
        boardNumbers[3].text = c.releases.ToString();
        boardNumbers[4].text = c.opens.ToString();
        boardNumbers[5].text = c.distance;
    }

    // The cue card and ghost hand while the laptop records
    bool Training => record != null && record.running && record.has_target;

    void ShowRecording()
    {
        bool running = record != null && record.running;
        if (running && tour.Running) tour.End();
        ShowControl(!running && controlFrame != null && controlFrame.driven ? controlFrame : null);
        if (running != card.Visible)
        {
            if (running) card.root.gameObject.SetActive(true);
            else card.Close();
        }
        ghost.Root.gameObject.SetActive(running && record.has_target);
        bool wristCue = running && record.wrist_cue;
        armShown = Mathf.MoveTowards(armShown, wristCue ? 1 : 0, Time.deltaTime / ArmFadeS);
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
            : record.seconds <= 0 ? "Continue when ready."
            : record.has_target ? "Do it with both hands." : "";
        if (record.has_target && record.target != null && record.target.Length == Sim.Joints)
        {
            float t = Mathf.Clamp01((Time.time - cueStartedAt) / FollowInS);
            for (int j = 0; j < Sim.Joints; j++) ghostAngles[j] = Mathf.Lerp(meta.rest[j], record.target[j], t);
            float wrist = Mathf.Lerp(0, record.wrist, t);
            ghost.Pose(ghostAngles);
            ghost.SetWrist(wrist, Mathf.SmoothStep(0, 1, armShown));
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
        heightText.text = handHeight ? "pinch to set"
            : $"{settings.deskHeight * 100:0} cm";
        sideText.text = settings.prostheticLeft ? "Left: the controller strapped to the left forearm"
            : "Right: the controller strapped to the right forearm";
        freeText.text = tracked.Tracked ? "Tracked from your real hand, nothing to set."
            : "Not seen. Hold it in view, or hold a controller in it.";
        recordText.text = record == null ? "Laptop not connected"
            : recording ? $"Cue {record.index + 1} of {record.count}" + (record.paused ? ", paused" : "")
            : "Not recording";
        recordError.text = record != null && !string.IsNullOrEmpty(record.error) ? $"<color=#ff3b3b>{record.error}</color>" : "";
        sessionButtons[0].text = record != null && record.paused ? "Continue" : "Pause";
        quitButton[0].text = Time.time - quitArmedAt <= QuitConfirmS ? "Press again to quit" : "Quit";
        var train = frame?.train;
        trainText.text = train == null ? "Laptop not connected"
            : !string.IsNullOrEmpty(train.status) ? train.status
            : !string.IsNullOrEmpty(train.session) ? "Fits the sign network to where the band sits now, on the session just recorded."
            : "Record a quick calibration, then calibrate the sign network on it here.";
        calibrations = train?.calibrations ?? new string[0];
        if (Array.IndexOf(calibrations, calPicked) < 0) calPicked = calibrations.Length > 0 ? calibrations[0] : null;
        int picked = Array.IndexOf(calibrations, calPicked);
        calText.text = calPicked == null ? "No calibrated network on the laptop yet."
            : $"{CalibrationText(calPicked)}{(picked == 0 ? ", the newest" : "")} ({picked + 1} of {calibrations.Length})";
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
        r.echo = link.Latest?.sent ?? 0;
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
        viewOut.panels = PanelsInView();
        viewOut.ray = rayOn ? rayOut : new float[0];
        viewOut.menu_pages = menu.PageNames;
        viewOut.free_tracked = tracked.Tracked;
        viewOut.free_angles = tracked.Angles;
        viewOut.other_tracked = otherHand.Tracked;
        viewOut.other_angles = otherHand.Angles;
        link.View = JsonUtility.ToJson(new ViewMessage { view = viewOut });
    }

    // Off the head the sim and link threads would keep running and drain the battery: they stop, and a timer closes
    // the app if it stays off. Killing the process is what the Quest does to a paused app anyway; settings are saved
    void OnApplicationPause(bool paused)
    {
        if (sim != null) sim.Paused = paused;
        if (paused)
        {
            settings.Save();
            link?.Stop();
            linkStopped = true;
            quitTimer?.Dispose();
            quitTimer = new System.Threading.Timer(_ => System.Diagnostics.Process.GetCurrentProcess().Kill(), null,
                QuitAfterOffMs, System.Threading.Timeout.Infinite);
            return;
        }
        quitTimer?.Dispose();
        quitTimer = null;
        if (!linkStopped) return;
        linkStopped = false;
        link = new Link();
        link.Start();
        sceneSent = -1;
    }

    void OnDestroy()
    {
        sim?.Stop();
        link?.Stop();
        quitTimer?.Dispose();
    }
}
