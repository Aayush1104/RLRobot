pub mod rmw {
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[link(name = "limo_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__limo_msgs__msg__LimoStatus() -> *const std::os::raw::c_void;
}

#[link(name = "limo_msgs__rosidl_generator_c")]
extern "C" {
    fn limo_msgs__msg__LimoStatus__init(msg: *mut LimoStatus) -> bool;
    fn limo_msgs__msg__LimoStatus__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<LimoStatus>, size: usize) -> bool;
    fn limo_msgs__msg__LimoStatus__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<LimoStatus>);
    fn limo_msgs__msg__LimoStatus__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<LimoStatus>, out_seq: *mut rosidl_runtime_rs::Sequence<LimoStatus>) -> bool;
}

// Corresponds to limo_msgs__msg__LimoStatus
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct LimoStatus {
    pub header: std_msgs::msg::rmw::Header,
    pub vehicle_state: u8,
    pub control_mode: u8,
    pub battery_voltage: f64,
    pub error_code: u16,
    pub motion_mode: u8,
}



impl Default for LimoStatus {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !limo_msgs__msg__LimoStatus__init(&mut msg as *mut _) {
        panic!("Call to limo_msgs__msg__LimoStatus__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for LimoStatus {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { limo_msgs__msg__LimoStatus__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { limo_msgs__msg__LimoStatus__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { limo_msgs__msg__LimoStatus__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for LimoStatus {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for LimoStatus where Self: Sized {
  const TYPE_NAME: &'static str = "limo_msgs/msg/LimoStatus";
  fn get_type_support() -> *const std::os::raw::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__limo_msgs__msg__LimoStatus() }
  }
}


}  // mod rmw


#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct LimoStatus {
    pub header: std_msgs::msg::Header,
    pub vehicle_state: u8,
    pub control_mode: u8,
    pub battery_voltage: f64,
    pub error_code: u16,
    pub motion_mode: u8,
}



impl Default for LimoStatus {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(crate::msg::rmw::LimoStatus::default())
  }
}

impl rosidl_runtime_rs::Message for LimoStatus {
  type RmwMsg = crate::msg::rmw::LimoStatus;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        vehicle_state: msg.vehicle_state,
        control_mode: msg.control_mode,
        battery_voltage: msg.battery_voltage,
        error_code: msg.error_code,
        motion_mode: msg.motion_mode,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
      vehicle_state: msg.vehicle_state,
      control_mode: msg.control_mode,
      battery_voltage: msg.battery_voltage,
      error_code: msg.error_code,
      motion_mode: msg.motion_mode,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      vehicle_state: msg.vehicle_state,
      control_mode: msg.control_mode,
      battery_voltage: msg.battery_voltage,
      error_code: msg.error_code,
      motion_mode: msg.motion_mode,
    }
  }
}


