using UnityEngine;

// Short sounds for the menu and the recording's cues, made at start so there are no audio files: a click when a press
// goes through, a softer tick when a hold starts and a faint one when the ray or a fingertip moves onto a control; for
// the cues (App.CueSounds) rising notes for a sign, falling ones for rest, a soft countdown pip, a low pause tone and
// three notes at the end. Quiet, and not placed in space: played as 3D sounds at the head they came out uneven, a
// sound inside the head has no side for the headset's spatial audio to put it on (Noah: inconsistent cue sounds).
public class Sounds
{
    public enum Kind { Click, Tick, Hover, Go, Rest, Count, Pause, Done }

    const int Rate = 48000;
    readonly AudioClip click, tick, hover, go, rest, count, pause, done;
    readonly AudioSource source;

    public Sounds()
    {
        source = new GameObject("sounds").AddComponent<AudioSource>();
        source.spatialBlend = 0;
        source.playOnAwake = false;
        Object.DontDestroyOnLoad(source.gameObject);
        // A click: a quick rising blip, two tones a fifth apart. A tick: one higher, shorter, softer tone
        click = Tone("click", 0.045f, 0.35f, (t, d) => Mathf.Lerp(880, 1320, t / d));
        tick = Tone("tick", 0.02f, 0.18f, (t, d) => 1760);
        hover = Tone("hover", 0.012f, 0.06f, (t, d) => 2640);
        go = Notes("go", 0.3f, 660, 990);
        rest = Notes("rest", 0.3f, 990, 660);
        count = Tone("count", 0.04f, 0.15f, (t, d) => 1320);
        pause = Tone("pause", 0.18f, 0.3f, (t, d) => 440);
        done = Notes("done", 0.3f, 660, 880, 1320);
    }

    public void Play(Kind kind, Vector3 at)
    {
        var clip = kind switch
        {
            Kind.Click => click, Kind.Tick => tick, Kind.Hover => hover, Kind.Go => go, Kind.Rest => rest,
            Kind.Count => count, Kind.Pause => pause, _ => done
        };
        source.PlayOneShot(clip);
    }

    // Notes one after another, 0.11 s apart, each fading on its own
    static AudioClip Notes(string name, float volume, params float[] hz)
    {
        const float Each = 0.11f;
        int step = Mathf.RoundToInt(Each * Rate);
        var data = new float[step * hz.Length + step];
        for (int k = 0; k < hz.Length; k++)
        {
            var note = Tone("", Each * 1.5f, volume, (t, d) => hz[k]);
            var part = new float[note.samples];
            note.GetData(part, 0);
            for (int i = 0; i < part.Length && k * step + i < data.Length; i++) data[k * step + i] += part[i];
        }
        var clip = AudioClip.Create(name, data.Length, 1, Rate, false);
        clip.SetData(data, 0);
        return clip;
    }

    static AudioClip Tone(string name, float seconds, float volume, System.Func<float, float, float> hz)
    {
        int n = Mathf.RoundToInt(seconds * Rate);
        var data = new float[n];
        double phase = 0;
        for (int i = 0; i < n; i++)
        {
            float t = (float)i / Rate;
            phase += 2 * Mathf.PI * hz(t, seconds) / Rate;
            // 3 ms in, then dying away, so it doesn't pop
            float env = Mathf.Min(1, t / 0.003f) * Mathf.Exp(-t / (seconds / 3));
            data[i] = volume * env * (float)System.Math.Sin(phase);
        }
        var clip = AudioClip.Create(name, n, 1, Rate, false);
        clip.SetData(data, 0);
        return clip;
    }
}
