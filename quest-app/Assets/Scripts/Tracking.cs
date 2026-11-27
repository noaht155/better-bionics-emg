using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.XR;

// What the headset tracks, for finding tracking trouble: each controller's state and how often and how long it lost
// tracking, each real hand, what the free hand goes by and what the adjusting ray does. Shown on the Setup page and
// sent with the view to the laptop's VR tab. Also written to the log under "[track]" (adb logcat -s Unity), so it
// can be followed over USB while someone wears the headset: every change, the devices coming and going, and once a
// second the raw state with positions
public class Tracking
{
    [Serializable]
    public class Side
    {
        // "tracked", "estimated" (position only estimated), "rotation only" (position lost, turning still followed),
        // "lost" or "off" (not connected or asleep)
        public string controller = "off";
        public bool hand;
        public int losses;
        // How long the current loss has lasted and how long the last one did (s)
        public float lost_s, last_loss_s;
        // From the tracked wrist to the controller, -1 without both (cm)
        public float wrist_cm = -1;
        // The device the hand's node resolves to and its raw flags, for the log
        public string device = "";
        [NonSerialized] public bool isTracked;
        [NonSerialized] public int flags;
        [NonSerialized] public Vector3 pos;
    }

    [Serializable]
    public class Out
    {
        public Side left = new Side(), right = new Side();
        // What the free hand goes by: "hand", "controller" or "none"
        public string free = "none";
        public string adjust = "", multimodal = "";
    }

    public readonly Out State = new Out();
    // More of the app's state for the once a second log line
    public string Extra = "";
    float leftLostAt = -1, rightLostAt = -1, loggedAt;
    string lastKey = "";
    readonly List<InputDevice> devices = new List<InputDevice>();
    readonly StringBuilder line = new StringBuilder();

    public Tracking()
    {
        InputDevices.deviceConnected += d => Debug.Log($"[track] {Time.time:0.00} connected {Describe(d)}");
        InputDevices.deviceDisconnected += d => Debug.Log($"[track] {Time.time:0.00} disconnected {Describe(d)}");
        InputDevices.deviceConfigChanged += d => Debug.Log($"[track] {Time.time:0.00} changed {Describe(d)}");
    }

    static string Describe(InputDevice d) => $"'{d.name}' ({d.characteristics})";

    // hasP, hasF: whether the app used the prosthetic and free controllers this frame
    public void Update(TrackedHand left, TrackedHand right, string free, string adjust, bool hasP, bool hasF)
    {
        Read(XRNode.LeftHand, left, State.left, ref leftLostAt);
        Read(XRNode.RightHand, right, State.right, ref rightLostAt);
        State.free = free;
        State.adjust = adjust;
        State.multimodal = Multimodal.Status;
        Log(hasP, hasF);
    }

    static void Read(XRNode node, TrackedHand hand, Side side, ref float lostAt)
    {
        var d = InputDevices.GetDeviceAtXRNode(node);
        string state = "off";
        Vector3 p = default;
        side.isTracked = false;
        side.flags = 0;
        if (d.isValid)
        {
            d.TryGetFeatureValue(CommonUsages.isTracked, out bool on);
            d.TryGetFeatureValue(CommonUsages.trackingState, out InputTrackingState flags);
            bool position = (flags & InputTrackingState.Position) != 0 && d.TryGetFeatureValue(CommonUsages.devicePosition, out p);
            state = position ? on ? "tracked" : "estimated" : (flags & InputTrackingState.Rotation) != 0 ? "rotation only" : "lost";
            side.isTracked = on;
            side.flags = (int)flags;
        }
        side.device = d.isValid ? d.name : "";
        side.pos = p;
        bool ok = state == "tracked";
        if (!ok && side.controller == "tracked")
        {
            side.losses++;
            lostAt = Time.time;
        }
        if (ok && lostAt >= 0)
        {
            side.last_loss_s = Time.time - lostAt;
            lostAt = -1;
        }
        side.lost_s = lostAt >= 0 ? Time.time - lostAt : 0;
        side.controller = state;
        side.hand = hand.Tracked;
        side.wrist_cm = (ok || state == "estimated") && hand.Tracked ? Vector3.Distance(hand.WristPos, p) * 100 : -1;
    }

    void Log(bool hasP, bool hasF)
    {
        string key = $"L {State.left.controller} '{State.left.device}' hand {State.left.hand} | " +
                     $"R {State.right.controller} '{State.right.device}' hand {State.right.hand} | " +
                     $"prosthetic used {hasP}, free controller used {hasF}, free by {State.free}, " +
                     $"multimodal {State.multimodal}, focused {Application.isFocused}";
        bool changed = key != lastKey;
        if (!changed && Time.time - loggedAt < 1) return;
        loggedAt = Time.time;
        lastKey = key;
        line.Clear();
        line.Append("[track] ").Append(Time.time.ToString("0.00")).Append(changed ? " CHANGE " : " ").Append(key);
        foreach (var (name, s) in new[] { ("L", State.left), ("R", State.right) })
            line.Append($" | {name} isTracked {s.isTracked} flags {s.flags} pos {s.pos.x:0.000} {s.pos.y:0.000} {s.pos.z:0.000}" +
                        $" wrist {s.wrist_cm:0}cm losses {s.losses}");
        if (!changed)
        {
            InputDevices.GetDevices(devices);
            line.Append(" | devices:");
            foreach (var d in devices) line.Append(' ').Append(Describe(d));
            line.Append($" | fps {1 / Mathf.Max(Time.smoothDeltaTime, 1e-4f):0}, ").Append(Extra);
        }
        Debug.Log(line.ToString());
    }

    public string Text() =>
        Line("Left", State.left) + "\n" + Line("Right", State.right) + "\nFree hand goes by: " + State.free
        + (State.adjust != "" ? "\n" + State.adjust : "");

    static string Line(string name, Side s) =>
        $"{name} controller {s.controller}" + (s.lost_s > 0 ? $" for {s.lost_s:0.0} s" : "")
        + $", lost {s.losses}x" + (s.losses > 0 ? $" (last {s.last_loss_s:0.0} s)" : "")
        + ", hand " + (s.hand ? "seen" + (s.wrist_cm >= 0 ? $", {s.wrist_cm:0} cm from it" : "") : "not seen");
}
