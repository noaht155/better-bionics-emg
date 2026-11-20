using UnityEngine;

// What the setup screen sets, kept on the headset (PlayerPrefs) between runs: which arm is the prosthetic one, each
// hand's offset from its controller, and where the desk stands in the room.
public class Settings
{
    // Offset of the wrist from the controller, in the controller's frame (m), and the hand's turn from the pose that
    // holds a controller naturally (degrees, x then y then z)
    public class Offset
    {
        public Vector3 pos, rot;
    }

    // Wrist about 9 cm behind and 3 cm below a Touch controller held normally
    public static readonly Vector3 HeldPos = new Vector3(0, -0.03f, -0.09f);

    public bool prostheticLeft = true;
    public Offset prosthetic = new Offset { pos = HeldPos }, free = new Offset { pos = HeldPos };
    public float deskHeight = 0.74f;
    public bool deskPlaced;
    public Vector3 deskPos;
    public float deskYaw;

    public static Settings Load()
    {
        var s = new Settings();
        s.prostheticLeft = PlayerPrefs.GetInt("prosthetic_left", 1) == 1;
        s.prosthetic = LoadOffset("prosthetic", s.prosthetic);
        s.free = LoadOffset("free", s.free);
        s.deskHeight = PlayerPrefs.GetFloat("desk_height", s.deskHeight);
        s.deskPlaced = PlayerPrefs.GetInt("desk_placed", 0) == 1;
        s.deskPos = LoadVector("desk_pos", Vector3.zero);
        s.deskYaw = PlayerPrefs.GetFloat("desk_yaw", 0);
        return s;
    }

    public void Save()
    {
        PlayerPrefs.SetInt("prosthetic_left", prostheticLeft ? 1 : 0);
        SaveOffset("prosthetic", prosthetic);
        SaveOffset("free", free);
        PlayerPrefs.SetFloat("desk_height", deskHeight);
        PlayerPrefs.SetInt("desk_placed", deskPlaced ? 1 : 0);
        SaveVector("desk_pos", deskPos);
        PlayerPrefs.SetFloat("desk_yaw", deskYaw);
        PlayerPrefs.Save();
    }

    static Offset LoadOffset(string key, Offset fallback) =>
        new Offset { pos = LoadVector(key + "_pos", fallback.pos), rot = LoadVector(key + "_rot", fallback.rot) };

    static void SaveOffset(string key, Offset o)
    {
        SaveVector(key + "_pos", o.pos);
        SaveVector(key + "_rot", o.rot);
    }

    static Vector3 LoadVector(string key, Vector3 fallback) => new Vector3(PlayerPrefs.GetFloat(key + "_x", fallback.x),
        PlayerPrefs.GetFloat(key + "_y", fallback.y), PlayerPrefs.GetFloat(key + "_z", fallback.z));

    static void SaveVector(string key, Vector3 v)
    {
        PlayerPrefs.SetFloat(key + "_x", v.x);
        PlayerPrefs.SetFloat(key + "_y", v.y);
        PlayerPrefs.SetFloat(key + "_z", v.z);
    }
}
