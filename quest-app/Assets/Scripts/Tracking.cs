using System;
using UnityEngine;
using UnityEngine.XR;

// What the headset tracks, for finding tracking trouble: each controller's state and how often and how long it lost
// tracking, each real hand, what the free hand goes by and what the adjusting ray does. Shown on the Setup page and
// sent with the view to the laptop's VR tab
public class Tracking
{
    [Serializable]
    public class Side
    {
        // "tracked", "rotation only" (position lost, turning still followed), "lost" or "off" (not connected or asleep)
        public string controller = "off";
        public bool hand;
        public int losses;
        // How long the current loss has lasted and how long the last one did (s)
        public float lost_s, last_loss_s;
        // From the tracked wrist to the controller, -1 without both (cm)
        public float wrist_cm = -1;
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
    float leftLostAt = -1, rightLostAt = -1;

    public void Update(TrackedHand left, TrackedHand right, string free, string adjust)
    {
        Read(XRNode.LeftHand, left, State.left, ref leftLostAt);
        Read(XRNode.RightHand, right, State.right, ref rightLostAt);
        State.free = free;
        State.adjust = adjust;
        State.multimodal = Multimodal.Status;
    }

    static void Read(XRNode node, TrackedHand hand, Side side, ref float lostAt)
    {
        var d = InputDevices.GetDeviceAtXRNode(node);
        string state = "off";
        Vector3 p = default;
        if (d.isValid)
        {
            d.TryGetFeatureValue(CommonUsages.isTracked, out bool on);
            d.TryGetFeatureValue(CommonUsages.trackingState, out InputTrackingState flags);
            bool position = (flags & InputTrackingState.Position) != 0 && d.TryGetFeatureValue(CommonUsages.devicePosition, out p);
            state = on && position ? "tracked" : (flags & InputTrackingState.Rotation) != 0 ? "rotation only" : "lost";
        }
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
        side.wrist_cm = ok && hand.Tracked ? Vector3.Distance(hand.WristPos, p) * 100 : -1;
    }

    public string Text() =>
        Line("Left", State.left) + "\n" + Line("Right", State.right) + "\nFree hand goes by: " + State.free
        + (State.adjust != "" ? "\n" + State.adjust : "");

    static string Line(string name, Side s) =>
        $"{name} controller {s.controller}" + (s.lost_s > 0 ? $" for {s.lost_s:0.0} s" : "")
        + $", lost {s.losses}x" + (s.losses > 0 ? $" (last {s.last_loss_s:0.0} s)" : "")
        + ", hand " + (s.hand ? "seen" + (s.wrist_cm >= 0 ? $", {s.wrist_cm:0} cm from it" : "") : "not seen");
}
