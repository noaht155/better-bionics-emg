using System.Collections.Generic;
using UnityEngine.InputSystem.Layouts;
using UnityEngine.InputSystem.XR;
using UnityEngine.XR;
using UnityEngine.XR.OpenXR;
using UnityEngine.XR.OpenXR.Features;
using UnityEngine.XR.OpenXR.Features.Interactions;
#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.XR.OpenXR.Features;
#endif

// The strapped controller's pose while no hand holds it. With hands and controllers tracked together (Multimodal),
// the headset reports a controller its own hand doesn't hold (XR_META_detached_controllers) under
// /user/detached_controller_meta/left or right instead of /user/hand/..., so the Touch profile's device stops
// tracking it: the controller strapped to the prosthetic arm is never held. This binds the detached paths (with the
// Touch Plus profile, so they don't clash with the Touch profile's bindings on the hand paths) as two more devices,
// found by name (Device) and not by XRNode, which keeps the held controllers on the hand nodes.
#if UNITY_EDITOR
[OpenXRFeature(UiName = "Detached controllers", BuildTargetGroups = new[] { BuildTargetGroup.Android },
    Company = "Better Bionics", Desc = "Tracks a controller that isn't held, such as one strapped to an arm.",
    OpenxrExtensionStrings = "XR_META_detached_controllers XR_META_touch_controller_plus", Version = "1.0.0",
    Category = FeatureCategory.Interaction, FeatureId = Id)]
#endif
public class DetachedControllers : OpenXRInteractionFeature
{
    public const string Id = "com.betterbionics.detachedcontrollers";
    const string Name = "Detached Touch Controller";
    const string Profile = MetaQuestTouchPlusControllerProfile.profile;
    const string LeftPath = "/user/detached_controller_meta/left", RightPath = "/user/detached_controller_meta/right";

    public static string Status { get; private set; } = "not started";
    static readonly List<InputDevice> found = new List<InputDevice>();

    // The detached controller on a side, if the runtime made the device
    public static InputDevice Device(bool left)
    {
        InputDevices.GetDevicesWithCharacteristics(
            InputDeviceCharacteristics.Controller | (left ? InputDeviceCharacteristics.Left : InputDeviceCharacteristics.Right), found);
        foreach (var d in found)
            if (d.name == Name)
                return d;
        return default;
    }

    protected override bool OnInstanceCreate(ulong instance)
    {
        if (!OpenXRRuntime.IsExtensionEnabled("XR_META_detached_controllers"))
        {
            Status = "not supported by this headset";
            return false;
        }
        if (!OpenXRRuntime.IsExtensionEnabled("XR_META_touch_controller_plus"))
        {
            Status = "Touch Plus profile not supported";
            return false;
        }
        Status = "on";
        return base.OnInstanceCreate(instance);
    }

    protected override void RegisterDeviceLayout()
    {
#if UNITY_EDITOR
        if (!OpenXRLoaderEnabledForSelectedBuildTarget(EditorUserBuildSettings.selectedBuildTargetGroup)) return;
#endif
        UnityEngine.InputSystem.InputSystem.RegisterLayout(typeof(MetaQuestTouchPlusControllerProfile.QuestTouchPlusController), "DetachedTouchController",
            new InputDeviceMatcher().WithInterface(XRUtilities.InterfaceMatchAnyVersion).WithProduct(Name));
    }

    protected override void UnregisterDeviceLayout()
    {
#if UNITY_EDITOR
        if (!OpenXRLoaderEnabledForSelectedBuildTarget(EditorUserBuildSettings.selectedBuildTargetGroup)) return;
#endif
        UnityEngine.InputSystem.InputSystem.RemoveLayout("DetachedTouchController");
    }

    protected override string GetDeviceLayoutName() => "DetachedTouchController";

    // Only the poses: the buttons of a controller nobody holds aren't needed
    protected override void RegisterActionMapsWithRuntime()
    {
        var side = InputDeviceCharacteristics.TrackedDevice | InputDeviceCharacteristics.Controller;
        AddActionMap(new ActionMapConfig
        {
            name = "detachedtouchcontroller",
            localizedName = Name,
            desiredInteractionProfile = Profile,
            manufacturer = "Oculus",
            serialNumber = "",
            deviceInfos = new List<DeviceConfig>
            {
                new DeviceConfig { characteristics = side | InputDeviceCharacteristics.Left, userPath = LeftPath },
                new DeviceConfig { characteristics = side | InputDeviceCharacteristics.Right, userPath = RightPath }
            },
            actions = new List<ActionConfig>
            {
                new ActionConfig
                {
                    name = "devicePose", localizedName = "Device Pose", type = ActionType.Pose,
                    usages = new List<string> { "Device" },
                    bindings = new List<ActionBinding>
                    {
                        new ActionBinding { interactionPath = MetaQuestTouchPlusControllerProfile.grip, interactionProfileName = Profile }
                    }
                }
            }
        });
    }
}
