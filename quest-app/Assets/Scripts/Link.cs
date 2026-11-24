using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

// The laptop over USB: the recorder app runs `adb reverse`, so the headset's localhost:8000 is the app's /ws/quest.
// It sends the control layer's output and the shared sim settings, and takes the grip force back for the motors.
// Without it the sim runs on its own with the hand at rest.
public class Link
{
    const string Url = "ws://127.0.0.1:8000/ws/quest";
    // The view for the laptop's VR tab goes out at about the rate the tab draws it
    const int SendMs = 33;

    [Serializable]
    public class Params
    {
        public string @object;
        public float mass_scale;
        public float max_force;
        public bool assist;
        public string rail;
    }

    [Serializable]
    public class Command
    {
        public int id;
        public string kind, arg;
    }

    // The recording on the laptop as the headset shows it (app.py headset_record_state)
    [Serializable]
    public class Record
    {
        public bool running, paused, hum, marked_bad, has_target;
        public int index, count;
        public string label, kind, text, next, error;
        public float seconds, elapsed, wrist;
        public bool wrist_cue;
        public float[] target;
    }

    [Serializable]
    public class Frame
    {
        public float[] angles;
        public float effort;
        public bool locked;
        public bool driven;
        public Params @params;
        public int params_seq;
        public Command[] commands;
        public Record record;
        public Haptics haptics;
        public Train train;
        public Control control;
    }

    // While control runs, what the board behind the desk shows (tools.Contact._quest_control)
    [Serializable]
    public class Control
    {
        public string label, state, grip;
        public float prob;
    }

    // Train in VR: the quick calibration on the last session (app.headset_train_state)
    [Serializable]
    public class Train
    {
        public string session, status, saved, error;
        public bool running;
        public string[] calibrations;
    }

    // What the laptop's motors play (app.headset_haptics_state)
    [Serializable]
    public class Haptics
    {
        public bool on, test;
        public string mode, surface;
        public float force, hz, friction, duty;
    }

    readonly CancellationTokenSource stop = new CancellationTokenSource();
    readonly ConcurrentQueue<string> outbox = new ConcurrentQueue<string>();
    volatile Frame latest;
    volatile bool connected;
    string lastProblem;
    // JSON sent every SendMs while connected, set by the app
    public volatile string Readout, View;
    // Sent once, the next time the loop sends (the scene for the VR tab)
    public volatile string Scene;

    public Frame Latest => latest;
    public bool Connected => connected;

    public void Start() => Task.Run(Run);

    public void Stop() => stop.Cancel();

    public void Send(string json) => outbox.Enqueue(json);

    async Task Run()
    {
        while (!stop.IsCancellationRequested)
        {
            using (var ws = new ClientWebSocket())
            {
                try
                {
                    await ws.ConnectAsync(new Uri(Url), stop.Token);
                    connected = true;
                    await Task.WhenAny(Receive(ws), SendLoop(ws));
                }
                catch (Exception e)
                {
                    // Logged once per change so adb logcat shows why the laptop isn't reached
                    string problem = e.GetType().Name + ": " + e.Message;
                    if (problem != lastProblem) Debug.Log("laptop link: " + problem);
                    lastProblem = problem;
                }
                connected = false;
                latest = null;
            }
            try { await Task.Delay(1000, stop.Token); }
            catch (TaskCanceledException) { }
        }
    }

    async Task Receive(ClientWebSocket ws)
    {
        var buf = new byte[1 << 14];
        var msg = new MemoryStream();
        while (ws.State == WebSocketState.Open)
        {
            msg.SetLength(0);
            WebSocketReceiveResult r;
            do
            {
                r = await ws.ReceiveAsync(new ArraySegment<byte>(buf), stop.Token);
                if (r.MessageType == WebSocketMessageType.Close) return;
                msg.Write(buf, 0, r.Count);
            } while (!r.EndOfMessage);
            var frame = JsonUtility.FromJson<Frame>(Encoding.UTF8.GetString(msg.GetBuffer(), 0, (int)msg.Length));
            if (frame?.angles != null && frame.angles.Length == Sim.Joints) latest = frame;
        }
    }

    async Task SendLoop(ClientWebSocket ws)
    {
        while (ws.State == WebSocketState.Open)
        {
            while (outbox.TryDequeue(out var json)) await SendText(ws, json);
            var scene = Scene;
            if (scene != null)
            {
                await SendText(ws, scene);
                if (Scene == scene) Scene = null;
            }
            var readout = Readout;
            if (readout != null) await SendText(ws, readout);
            var view = View;
            if (view != null) await SendText(ws, view);
            await Task.Delay(SendMs, stop.Token);
        }
    }

    Task SendText(ClientWebSocket ws, string json) =>
        ws.SendAsync(new ArraySegment<byte>(Encoding.UTF8.GetBytes(json)), WebSocketMessageType.Text, true, stop.Token);
}
