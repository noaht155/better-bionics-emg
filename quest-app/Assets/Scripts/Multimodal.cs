using System;
using System.Runtime.InteropServices;
using UnityEngine.XR.OpenXR;
using UnityEngine.XR.OpenXR.Features;
#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.XR.OpenXR.Features;
#endif

// Quest 3 "multimodal": the hand without a controller stays hand tracked while the other hand holds one (the
// controller strapped to the prosthetic arm). Without XR_META_simultaneous_hands_and_controllers the headset uses
// either hands or controllers, and holding one controller turns hand tracking off for both. Meta's SDK does this
// through its own camera rig; this feature only asks for the extension and turns it on when the session begins.
#if UNITY_EDITOR
[OpenXRFeature(UiName = "Simultaneous hands and controllers", BuildTargetGroups = new[] { BuildTargetGroup.Android },
    Company = "Better Bionics", Desc = "Hand tracking on one hand while the other holds a controller.",
    OpenxrExtensionStrings = Extension, Version = "1.0.0", FeatureId = Id)]
#endif
public class Multimodal : OpenXRFeature
{
    public const string Id = "com.betterbionics.multimodal";
    const string Extension = "XR_META_simultaneous_hands_and_controllers";
    // XR_TYPE_SIMULTANEOUS_HANDS_AND_CONTROLLERS_TRACKING_RESUME_INFO_META in openxr.h
    const int ResumeInfoType = 1000532002;

    [StructLayout(LayoutKind.Sequential)]
    struct ResumeInfo
    {
        public int type;
        public IntPtr next;
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    delegate int GetInstanceProcAddr(ulong instance, [MarshalAs(UnmanagedType.LPStr)] string name, out IntPtr function);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    delegate int ResumeTracking(ulong session, ref ResumeInfo info);

    ResumeTracking resume;

    // What happened, for the menu's status line
    public static string Status { get; private set; } = "not started";

    protected override bool OnInstanceCreate(ulong instance)
    {
        if (!OpenXRRuntime.IsExtensionEnabled(Extension))
        {
            Status = "not supported by this headset";
            return true;
        }
        var getProc = Marshal.GetDelegateForFunctionPointer<GetInstanceProcAddr>(xrGetInstanceProcAddr);
        int result = getProc(instance, "xrResumeSimultaneousHandsAndControllersTrackingMETA", out var function);
        if (result == 0 && function != IntPtr.Zero) resume = Marshal.GetDelegateForFunctionPointer<ResumeTracking>(function);
        else Status = "resume function missing (" + result + ")";
        return true;
    }

    // The runtime turns it off with every new session
    protected override void OnSessionBegin(ulong session)
    {
        if (resume == null) return;
        var info = new ResumeInfo { type = ResumeInfoType };
        int result = resume(session, ref info);
        Status = result == 0 ? "on" : "resume failed (" + result + ")";
    }
}
