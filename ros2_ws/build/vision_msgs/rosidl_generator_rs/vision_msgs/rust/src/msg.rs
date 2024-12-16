pub mod rmw {
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[link(name = "vision_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__BoundingBox2D() -> *const std::os::raw::c_void;
}

#[link(name = "vision_msgs__rosidl_generator_c")]
extern "C" {
    fn vision_msgs__msg__BoundingBox2D__init(msg: *mut BoundingBox2D) -> bool;
    fn vision_msgs__msg__BoundingBox2D__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<BoundingBox2D>, size: usize) -> bool;
    fn vision_msgs__msg__BoundingBox2D__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<BoundingBox2D>);
    fn vision_msgs__msg__BoundingBox2D__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<BoundingBox2D>, out_seq: *mut rosidl_runtime_rs::Sequence<BoundingBox2D>) -> bool;
}

// Corresponds to vision_msgs__msg__BoundingBox2D
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct BoundingBox2D {
    pub center: crate::msg::rmw::Pose2D,
    pub size_x: f64,
    pub size_y: f64,
}



impl Default for BoundingBox2D {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !vision_msgs__msg__BoundingBox2D__init(&mut msg as *mut _) {
        panic!("Call to vision_msgs__msg__BoundingBox2D__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for BoundingBox2D {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__BoundingBox2D__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__BoundingBox2D__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__BoundingBox2D__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for BoundingBox2D {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for BoundingBox2D where Self: Sized {
  const TYPE_NAME: &'static str = "vision_msgs/msg/BoundingBox2D";
  fn get_type_support() -> *const std::os::raw::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__BoundingBox2D() }
  }
}


#[link(name = "vision_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__BoundingBox2DArray() -> *const std::os::raw::c_void;
}

#[link(name = "vision_msgs__rosidl_generator_c")]
extern "C" {
    fn vision_msgs__msg__BoundingBox2DArray__init(msg: *mut BoundingBox2DArray) -> bool;
    fn vision_msgs__msg__BoundingBox2DArray__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<BoundingBox2DArray>, size: usize) -> bool;
    fn vision_msgs__msg__BoundingBox2DArray__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<BoundingBox2DArray>);
    fn vision_msgs__msg__BoundingBox2DArray__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<BoundingBox2DArray>, out_seq: *mut rosidl_runtime_rs::Sequence<BoundingBox2DArray>) -> bool;
}

// Corresponds to vision_msgs__msg__BoundingBox2DArray
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct BoundingBox2DArray {
    pub header: std_msgs::msg::rmw::Header,
    pub boxes: rosidl_runtime_rs::Sequence<crate::msg::rmw::BoundingBox2D>,
}



impl Default for BoundingBox2DArray {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !vision_msgs__msg__BoundingBox2DArray__init(&mut msg as *mut _) {
        panic!("Call to vision_msgs__msg__BoundingBox2DArray__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for BoundingBox2DArray {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__BoundingBox2DArray__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__BoundingBox2DArray__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__BoundingBox2DArray__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for BoundingBox2DArray {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for BoundingBox2DArray where Self: Sized {
  const TYPE_NAME: &'static str = "vision_msgs/msg/BoundingBox2DArray";
  fn get_type_support() -> *const std::os::raw::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__BoundingBox2DArray() }
  }
}


#[link(name = "vision_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__BoundingBox3D() -> *const std::os::raw::c_void;
}

#[link(name = "vision_msgs__rosidl_generator_c")]
extern "C" {
    fn vision_msgs__msg__BoundingBox3D__init(msg: *mut BoundingBox3D) -> bool;
    fn vision_msgs__msg__BoundingBox3D__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<BoundingBox3D>, size: usize) -> bool;
    fn vision_msgs__msg__BoundingBox3D__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<BoundingBox3D>);
    fn vision_msgs__msg__BoundingBox3D__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<BoundingBox3D>, out_seq: *mut rosidl_runtime_rs::Sequence<BoundingBox3D>) -> bool;
}

// Corresponds to vision_msgs__msg__BoundingBox3D
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct BoundingBox3D {
    pub center: geometry_msgs::msg::rmw::Pose,
    pub size: geometry_msgs::msg::rmw::Vector3,
}



impl Default for BoundingBox3D {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !vision_msgs__msg__BoundingBox3D__init(&mut msg as *mut _) {
        panic!("Call to vision_msgs__msg__BoundingBox3D__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for BoundingBox3D {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__BoundingBox3D__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__BoundingBox3D__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__BoundingBox3D__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for BoundingBox3D {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for BoundingBox3D where Self: Sized {
  const TYPE_NAME: &'static str = "vision_msgs/msg/BoundingBox3D";
  fn get_type_support() -> *const std::os::raw::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__BoundingBox3D() }
  }
}


#[link(name = "vision_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__BoundingBox3DArray() -> *const std::os::raw::c_void;
}

#[link(name = "vision_msgs__rosidl_generator_c")]
extern "C" {
    fn vision_msgs__msg__BoundingBox3DArray__init(msg: *mut BoundingBox3DArray) -> bool;
    fn vision_msgs__msg__BoundingBox3DArray__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<BoundingBox3DArray>, size: usize) -> bool;
    fn vision_msgs__msg__BoundingBox3DArray__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<BoundingBox3DArray>);
    fn vision_msgs__msg__BoundingBox3DArray__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<BoundingBox3DArray>, out_seq: *mut rosidl_runtime_rs::Sequence<BoundingBox3DArray>) -> bool;
}

// Corresponds to vision_msgs__msg__BoundingBox3DArray
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct BoundingBox3DArray {
    pub header: std_msgs::msg::rmw::Header,
    pub boxes: rosidl_runtime_rs::Sequence<crate::msg::rmw::BoundingBox3D>,
}



impl Default for BoundingBox3DArray {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !vision_msgs__msg__BoundingBox3DArray__init(&mut msg as *mut _) {
        panic!("Call to vision_msgs__msg__BoundingBox3DArray__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for BoundingBox3DArray {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__BoundingBox3DArray__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__BoundingBox3DArray__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__BoundingBox3DArray__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for BoundingBox3DArray {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for BoundingBox3DArray where Self: Sized {
  const TYPE_NAME: &'static str = "vision_msgs/msg/BoundingBox3DArray";
  fn get_type_support() -> *const std::os::raw::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__BoundingBox3DArray() }
  }
}


#[link(name = "vision_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__Classification() -> *const std::os::raw::c_void;
}

#[link(name = "vision_msgs__rosidl_generator_c")]
extern "C" {
    fn vision_msgs__msg__Classification__init(msg: *mut Classification) -> bool;
    fn vision_msgs__msg__Classification__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<Classification>, size: usize) -> bool;
    fn vision_msgs__msg__Classification__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Classification>);
    fn vision_msgs__msg__Classification__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<Classification>, out_seq: *mut rosidl_runtime_rs::Sequence<Classification>) -> bool;
}

// Corresponds to vision_msgs__msg__Classification
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Classification {
    pub header: std_msgs::msg::rmw::Header,
    pub results: rosidl_runtime_rs::Sequence<crate::msg::rmw::ObjectHypothesis>,
}



impl Default for Classification {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !vision_msgs__msg__Classification__init(&mut msg as *mut _) {
        panic!("Call to vision_msgs__msg__Classification__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for Classification {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__Classification__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__Classification__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__Classification__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for Classification {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for Classification where Self: Sized {
  const TYPE_NAME: &'static str = "vision_msgs/msg/Classification";
  fn get_type_support() -> *const std::os::raw::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__Classification() }
  }
}


#[link(name = "vision_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__Detection2DArray() -> *const std::os::raw::c_void;
}

#[link(name = "vision_msgs__rosidl_generator_c")]
extern "C" {
    fn vision_msgs__msg__Detection2DArray__init(msg: *mut Detection2DArray) -> bool;
    fn vision_msgs__msg__Detection2DArray__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<Detection2DArray>, size: usize) -> bool;
    fn vision_msgs__msg__Detection2DArray__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Detection2DArray>);
    fn vision_msgs__msg__Detection2DArray__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<Detection2DArray>, out_seq: *mut rosidl_runtime_rs::Sequence<Detection2DArray>) -> bool;
}

// Corresponds to vision_msgs__msg__Detection2DArray
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Detection2DArray {
    pub header: std_msgs::msg::rmw::Header,
    pub detections: rosidl_runtime_rs::Sequence<crate::msg::rmw::Detection2D>,
}



impl Default for Detection2DArray {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !vision_msgs__msg__Detection2DArray__init(&mut msg as *mut _) {
        panic!("Call to vision_msgs__msg__Detection2DArray__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for Detection2DArray {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__Detection2DArray__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__Detection2DArray__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__Detection2DArray__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for Detection2DArray {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for Detection2DArray where Self: Sized {
  const TYPE_NAME: &'static str = "vision_msgs/msg/Detection2DArray";
  fn get_type_support() -> *const std::os::raw::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__Detection2DArray() }
  }
}


#[link(name = "vision_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__Detection2D() -> *const std::os::raw::c_void;
}

#[link(name = "vision_msgs__rosidl_generator_c")]
extern "C" {
    fn vision_msgs__msg__Detection2D__init(msg: *mut Detection2D) -> bool;
    fn vision_msgs__msg__Detection2D__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<Detection2D>, size: usize) -> bool;
    fn vision_msgs__msg__Detection2D__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Detection2D>);
    fn vision_msgs__msg__Detection2D__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<Detection2D>, out_seq: *mut rosidl_runtime_rs::Sequence<Detection2D>) -> bool;
}

// Corresponds to vision_msgs__msg__Detection2D
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Detection2D {
    pub header: std_msgs::msg::rmw::Header,
    pub results: rosidl_runtime_rs::Sequence<crate::msg::rmw::ObjectHypothesisWithPose>,
    pub bbox: crate::msg::rmw::BoundingBox2D,
    pub id: rosidl_runtime_rs::String,
}



impl Default for Detection2D {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !vision_msgs__msg__Detection2D__init(&mut msg as *mut _) {
        panic!("Call to vision_msgs__msg__Detection2D__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for Detection2D {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__Detection2D__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__Detection2D__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__Detection2D__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for Detection2D {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for Detection2D where Self: Sized {
  const TYPE_NAME: &'static str = "vision_msgs/msg/Detection2D";
  fn get_type_support() -> *const std::os::raw::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__Detection2D() }
  }
}


#[link(name = "vision_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__Detection3DArray() -> *const std::os::raw::c_void;
}

#[link(name = "vision_msgs__rosidl_generator_c")]
extern "C" {
    fn vision_msgs__msg__Detection3DArray__init(msg: *mut Detection3DArray) -> bool;
    fn vision_msgs__msg__Detection3DArray__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<Detection3DArray>, size: usize) -> bool;
    fn vision_msgs__msg__Detection3DArray__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Detection3DArray>);
    fn vision_msgs__msg__Detection3DArray__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<Detection3DArray>, out_seq: *mut rosidl_runtime_rs::Sequence<Detection3DArray>) -> bool;
}

// Corresponds to vision_msgs__msg__Detection3DArray
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Detection3DArray {
    pub header: std_msgs::msg::rmw::Header,
    pub detections: rosidl_runtime_rs::Sequence<crate::msg::rmw::Detection3D>,
}



impl Default for Detection3DArray {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !vision_msgs__msg__Detection3DArray__init(&mut msg as *mut _) {
        panic!("Call to vision_msgs__msg__Detection3DArray__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for Detection3DArray {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__Detection3DArray__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__Detection3DArray__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__Detection3DArray__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for Detection3DArray {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for Detection3DArray where Self: Sized {
  const TYPE_NAME: &'static str = "vision_msgs/msg/Detection3DArray";
  fn get_type_support() -> *const std::os::raw::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__Detection3DArray() }
  }
}


#[link(name = "vision_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__Detection3D() -> *const std::os::raw::c_void;
}

#[link(name = "vision_msgs__rosidl_generator_c")]
extern "C" {
    fn vision_msgs__msg__Detection3D__init(msg: *mut Detection3D) -> bool;
    fn vision_msgs__msg__Detection3D__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<Detection3D>, size: usize) -> bool;
    fn vision_msgs__msg__Detection3D__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Detection3D>);
    fn vision_msgs__msg__Detection3D__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<Detection3D>, out_seq: *mut rosidl_runtime_rs::Sequence<Detection3D>) -> bool;
}

// Corresponds to vision_msgs__msg__Detection3D
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Detection3D {
    pub header: std_msgs::msg::rmw::Header,
    pub results: rosidl_runtime_rs::Sequence<crate::msg::rmw::ObjectHypothesisWithPose>,
    pub bbox: crate::msg::rmw::BoundingBox3D,
    pub id: rosidl_runtime_rs::String,
}



impl Default for Detection3D {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !vision_msgs__msg__Detection3D__init(&mut msg as *mut _) {
        panic!("Call to vision_msgs__msg__Detection3D__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for Detection3D {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__Detection3D__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__Detection3D__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__Detection3D__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for Detection3D {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for Detection3D where Self: Sized {
  const TYPE_NAME: &'static str = "vision_msgs/msg/Detection3D";
  fn get_type_support() -> *const std::os::raw::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__Detection3D() }
  }
}


#[link(name = "vision_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__LabelInfo() -> *const std::os::raw::c_void;
}

#[link(name = "vision_msgs__rosidl_generator_c")]
extern "C" {
    fn vision_msgs__msg__LabelInfo__init(msg: *mut LabelInfo) -> bool;
    fn vision_msgs__msg__LabelInfo__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<LabelInfo>, size: usize) -> bool;
    fn vision_msgs__msg__LabelInfo__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<LabelInfo>);
    fn vision_msgs__msg__LabelInfo__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<LabelInfo>, out_seq: *mut rosidl_runtime_rs::Sequence<LabelInfo>) -> bool;
}

// Corresponds to vision_msgs__msg__LabelInfo
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct LabelInfo {
    pub header: std_msgs::msg::rmw::Header,
    pub class_map: rosidl_runtime_rs::Sequence<crate::msg::rmw::VisionClass>,
    pub threshold: f32,
}



impl Default for LabelInfo {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !vision_msgs__msg__LabelInfo__init(&mut msg as *mut _) {
        panic!("Call to vision_msgs__msg__LabelInfo__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for LabelInfo {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__LabelInfo__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__LabelInfo__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__LabelInfo__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for LabelInfo {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for LabelInfo where Self: Sized {
  const TYPE_NAME: &'static str = "vision_msgs/msg/LabelInfo";
  fn get_type_support() -> *const std::os::raw::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__LabelInfo() }
  }
}


#[link(name = "vision_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__ObjectHypothesis() -> *const std::os::raw::c_void;
}

#[link(name = "vision_msgs__rosidl_generator_c")]
extern "C" {
    fn vision_msgs__msg__ObjectHypothesis__init(msg: *mut ObjectHypothesis) -> bool;
    fn vision_msgs__msg__ObjectHypothesis__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<ObjectHypothesis>, size: usize) -> bool;
    fn vision_msgs__msg__ObjectHypothesis__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<ObjectHypothesis>);
    fn vision_msgs__msg__ObjectHypothesis__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<ObjectHypothesis>, out_seq: *mut rosidl_runtime_rs::Sequence<ObjectHypothesis>) -> bool;
}

// Corresponds to vision_msgs__msg__ObjectHypothesis
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct ObjectHypothesis {
    pub class_id: rosidl_runtime_rs::String,
    pub score: f64,
}



impl Default for ObjectHypothesis {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !vision_msgs__msg__ObjectHypothesis__init(&mut msg as *mut _) {
        panic!("Call to vision_msgs__msg__ObjectHypothesis__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for ObjectHypothesis {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__ObjectHypothesis__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__ObjectHypothesis__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__ObjectHypothesis__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for ObjectHypothesis {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for ObjectHypothesis where Self: Sized {
  const TYPE_NAME: &'static str = "vision_msgs/msg/ObjectHypothesis";
  fn get_type_support() -> *const std::os::raw::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__ObjectHypothesis() }
  }
}


#[link(name = "vision_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__ObjectHypothesisWithPose() -> *const std::os::raw::c_void;
}

#[link(name = "vision_msgs__rosidl_generator_c")]
extern "C" {
    fn vision_msgs__msg__ObjectHypothesisWithPose__init(msg: *mut ObjectHypothesisWithPose) -> bool;
    fn vision_msgs__msg__ObjectHypothesisWithPose__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<ObjectHypothesisWithPose>, size: usize) -> bool;
    fn vision_msgs__msg__ObjectHypothesisWithPose__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<ObjectHypothesisWithPose>);
    fn vision_msgs__msg__ObjectHypothesisWithPose__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<ObjectHypothesisWithPose>, out_seq: *mut rosidl_runtime_rs::Sequence<ObjectHypothesisWithPose>) -> bool;
}

// Corresponds to vision_msgs__msg__ObjectHypothesisWithPose
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct ObjectHypothesisWithPose {
    pub hypothesis: crate::msg::rmw::ObjectHypothesis,
    pub pose: geometry_msgs::msg::rmw::PoseWithCovariance,
}



impl Default for ObjectHypothesisWithPose {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !vision_msgs__msg__ObjectHypothesisWithPose__init(&mut msg as *mut _) {
        panic!("Call to vision_msgs__msg__ObjectHypothesisWithPose__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for ObjectHypothesisWithPose {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__ObjectHypothesisWithPose__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__ObjectHypothesisWithPose__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__ObjectHypothesisWithPose__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for ObjectHypothesisWithPose {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for ObjectHypothesisWithPose where Self: Sized {
  const TYPE_NAME: &'static str = "vision_msgs/msg/ObjectHypothesisWithPose";
  fn get_type_support() -> *const std::os::raw::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__ObjectHypothesisWithPose() }
  }
}


#[link(name = "vision_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__VisionClass() -> *const std::os::raw::c_void;
}

#[link(name = "vision_msgs__rosidl_generator_c")]
extern "C" {
    fn vision_msgs__msg__VisionClass__init(msg: *mut VisionClass) -> bool;
    fn vision_msgs__msg__VisionClass__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<VisionClass>, size: usize) -> bool;
    fn vision_msgs__msg__VisionClass__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<VisionClass>);
    fn vision_msgs__msg__VisionClass__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<VisionClass>, out_seq: *mut rosidl_runtime_rs::Sequence<VisionClass>) -> bool;
}

// Corresponds to vision_msgs__msg__VisionClass
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct VisionClass {
    pub class_id: u16,
    pub class_name: rosidl_runtime_rs::String,
}



impl Default for VisionClass {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !vision_msgs__msg__VisionClass__init(&mut msg as *mut _) {
        panic!("Call to vision_msgs__msg__VisionClass__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for VisionClass {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__VisionClass__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__VisionClass__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__VisionClass__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for VisionClass {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for VisionClass where Self: Sized {
  const TYPE_NAME: &'static str = "vision_msgs/msg/VisionClass";
  fn get_type_support() -> *const std::os::raw::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__VisionClass() }
  }
}


#[link(name = "vision_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__Point2D() -> *const std::os::raw::c_void;
}

#[link(name = "vision_msgs__rosidl_generator_c")]
extern "C" {
    fn vision_msgs__msg__Point2D__init(msg: *mut Point2D) -> bool;
    fn vision_msgs__msg__Point2D__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<Point2D>, size: usize) -> bool;
    fn vision_msgs__msg__Point2D__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Point2D>);
    fn vision_msgs__msg__Point2D__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<Point2D>, out_seq: *mut rosidl_runtime_rs::Sequence<Point2D>) -> bool;
}

// Corresponds to vision_msgs__msg__Point2D
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}



impl Default for Point2D {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !vision_msgs__msg__Point2D__init(&mut msg as *mut _) {
        panic!("Call to vision_msgs__msg__Point2D__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for Point2D {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__Point2D__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__Point2D__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__Point2D__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for Point2D {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for Point2D where Self: Sized {
  const TYPE_NAME: &'static str = "vision_msgs/msg/Point2D";
  fn get_type_support() -> *const std::os::raw::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__Point2D() }
  }
}


#[link(name = "vision_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__Pose2D() -> *const std::os::raw::c_void;
}

#[link(name = "vision_msgs__rosidl_generator_c")]
extern "C" {
    fn vision_msgs__msg__Pose2D__init(msg: *mut Pose2D) -> bool;
    fn vision_msgs__msg__Pose2D__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<Pose2D>, size: usize) -> bool;
    fn vision_msgs__msg__Pose2D__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Pose2D>);
    fn vision_msgs__msg__Pose2D__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<Pose2D>, out_seq: *mut rosidl_runtime_rs::Sequence<Pose2D>) -> bool;
}

// Corresponds to vision_msgs__msg__Pose2D
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Pose2D {
    pub position: crate::msg::rmw::Point2D,
    pub theta: f64,
}



impl Default for Pose2D {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !vision_msgs__msg__Pose2D__init(&mut msg as *mut _) {
        panic!("Call to vision_msgs__msg__Pose2D__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for Pose2D {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__Pose2D__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__Pose2D__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__Pose2D__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for Pose2D {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for Pose2D where Self: Sized {
  const TYPE_NAME: &'static str = "vision_msgs/msg/Pose2D";
  fn get_type_support() -> *const std::os::raw::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__Pose2D() }
  }
}


#[link(name = "vision_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__VisionInfo() -> *const std::os::raw::c_void;
}

#[link(name = "vision_msgs__rosidl_generator_c")]
extern "C" {
    fn vision_msgs__msg__VisionInfo__init(msg: *mut VisionInfo) -> bool;
    fn vision_msgs__msg__VisionInfo__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<VisionInfo>, size: usize) -> bool;
    fn vision_msgs__msg__VisionInfo__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<VisionInfo>);
    fn vision_msgs__msg__VisionInfo__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<VisionInfo>, out_seq: *mut rosidl_runtime_rs::Sequence<VisionInfo>) -> bool;
}

// Corresponds to vision_msgs__msg__VisionInfo
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct VisionInfo {
    pub header: std_msgs::msg::rmw::Header,
    pub method: rosidl_runtime_rs::String,
    pub database_location: rosidl_runtime_rs::String,
    pub database_version: i32,
}



impl Default for VisionInfo {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !vision_msgs__msg__VisionInfo__init(&mut msg as *mut _) {
        panic!("Call to vision_msgs__msg__VisionInfo__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for VisionInfo {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__VisionInfo__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__VisionInfo__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vision_msgs__msg__VisionInfo__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for VisionInfo {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for VisionInfo where Self: Sized {
  const TYPE_NAME: &'static str = "vision_msgs/msg/VisionInfo";
  fn get_type_support() -> *const std::os::raw::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__vision_msgs__msg__VisionInfo() }
  }
}


}  // mod rmw


#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct BoundingBox2D {
    pub center: crate::msg::Pose2D,
    pub size_x: f64,
    pub size_y: f64,
}



impl Default for BoundingBox2D {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(crate::msg::rmw::BoundingBox2D::default())
  }
}

impl rosidl_runtime_rs::Message for BoundingBox2D {
  type RmwMsg = crate::msg::rmw::BoundingBox2D;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        center: crate::msg::Pose2D::into_rmw_message(std::borrow::Cow::Owned(msg.center)).into_owned(),
        size_x: msg.size_x,
        size_y: msg.size_y,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        center: crate::msg::Pose2D::into_rmw_message(std::borrow::Cow::Borrowed(&msg.center)).into_owned(),
      size_x: msg.size_x,
      size_y: msg.size_y,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      center: crate::msg::Pose2D::from_rmw_message(msg.center),
      size_x: msg.size_x,
      size_y: msg.size_y,
    }
  }
}


#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct BoundingBox2DArray {
    pub header: std_msgs::msg::Header,
    pub boxes: Vec<crate::msg::BoundingBox2D>,
}



impl Default for BoundingBox2DArray {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(crate::msg::rmw::BoundingBox2DArray::default())
  }
}

impl rosidl_runtime_rs::Message for BoundingBox2DArray {
  type RmwMsg = crate::msg::rmw::BoundingBox2DArray;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        boxes: msg.boxes
          .into_iter()
          .map(|elem| crate::msg::BoundingBox2D::into_rmw_message(std::borrow::Cow::Owned(elem)).into_owned())
          .collect(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        boxes: msg.boxes
          .iter()
          .map(|elem| crate::msg::BoundingBox2D::into_rmw_message(std::borrow::Cow::Borrowed(elem)).into_owned())
          .collect(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      boxes: msg.boxes
          .into_iter()
          .map(crate::msg::BoundingBox2D::from_rmw_message)
          .collect(),
    }
  }
}


#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct BoundingBox3D {
    pub center: geometry_msgs::msg::Pose,
    pub size: geometry_msgs::msg::Vector3,
}



impl Default for BoundingBox3D {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(crate::msg::rmw::BoundingBox3D::default())
  }
}

impl rosidl_runtime_rs::Message for BoundingBox3D {
  type RmwMsg = crate::msg::rmw::BoundingBox3D;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        center: geometry_msgs::msg::Pose::into_rmw_message(std::borrow::Cow::Owned(msg.center)).into_owned(),
        size: geometry_msgs::msg::Vector3::into_rmw_message(std::borrow::Cow::Owned(msg.size)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        center: geometry_msgs::msg::Pose::into_rmw_message(std::borrow::Cow::Borrowed(&msg.center)).into_owned(),
        size: geometry_msgs::msg::Vector3::into_rmw_message(std::borrow::Cow::Borrowed(&msg.size)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      center: geometry_msgs::msg::Pose::from_rmw_message(msg.center),
      size: geometry_msgs::msg::Vector3::from_rmw_message(msg.size),
    }
  }
}


#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct BoundingBox3DArray {
    pub header: std_msgs::msg::Header,
    pub boxes: Vec<crate::msg::BoundingBox3D>,
}



impl Default for BoundingBox3DArray {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(crate::msg::rmw::BoundingBox3DArray::default())
  }
}

impl rosidl_runtime_rs::Message for BoundingBox3DArray {
  type RmwMsg = crate::msg::rmw::BoundingBox3DArray;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        boxes: msg.boxes
          .into_iter()
          .map(|elem| crate::msg::BoundingBox3D::into_rmw_message(std::borrow::Cow::Owned(elem)).into_owned())
          .collect(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        boxes: msg.boxes
          .iter()
          .map(|elem| crate::msg::BoundingBox3D::into_rmw_message(std::borrow::Cow::Borrowed(elem)).into_owned())
          .collect(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      boxes: msg.boxes
          .into_iter()
          .map(crate::msg::BoundingBox3D::from_rmw_message)
          .collect(),
    }
  }
}


#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Classification {
    pub header: std_msgs::msg::Header,
    pub results: Vec<crate::msg::ObjectHypothesis>,
}



impl Default for Classification {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(crate::msg::rmw::Classification::default())
  }
}

impl rosidl_runtime_rs::Message for Classification {
  type RmwMsg = crate::msg::rmw::Classification;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        results: msg.results
          .into_iter()
          .map(|elem| crate::msg::ObjectHypothesis::into_rmw_message(std::borrow::Cow::Owned(elem)).into_owned())
          .collect(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        results: msg.results
          .iter()
          .map(|elem| crate::msg::ObjectHypothesis::into_rmw_message(std::borrow::Cow::Borrowed(elem)).into_owned())
          .collect(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      results: msg.results
          .into_iter()
          .map(crate::msg::ObjectHypothesis::from_rmw_message)
          .collect(),
    }
  }
}


#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Detection2DArray {
    pub header: std_msgs::msg::Header,
    pub detections: Vec<crate::msg::Detection2D>,
}



impl Default for Detection2DArray {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(crate::msg::rmw::Detection2DArray::default())
  }
}

impl rosidl_runtime_rs::Message for Detection2DArray {
  type RmwMsg = crate::msg::rmw::Detection2DArray;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        detections: msg.detections
          .into_iter()
          .map(|elem| crate::msg::Detection2D::into_rmw_message(std::borrow::Cow::Owned(elem)).into_owned())
          .collect(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        detections: msg.detections
          .iter()
          .map(|elem| crate::msg::Detection2D::into_rmw_message(std::borrow::Cow::Borrowed(elem)).into_owned())
          .collect(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      detections: msg.detections
          .into_iter()
          .map(crate::msg::Detection2D::from_rmw_message)
          .collect(),
    }
  }
}


#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Detection2D {
    pub header: std_msgs::msg::Header,
    pub results: Vec<crate::msg::ObjectHypothesisWithPose>,
    pub bbox: crate::msg::BoundingBox2D,
    pub id: std::string::String,
}



impl Default for Detection2D {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(crate::msg::rmw::Detection2D::default())
  }
}

impl rosidl_runtime_rs::Message for Detection2D {
  type RmwMsg = crate::msg::rmw::Detection2D;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        results: msg.results
          .into_iter()
          .map(|elem| crate::msg::ObjectHypothesisWithPose::into_rmw_message(std::borrow::Cow::Owned(elem)).into_owned())
          .collect(),
        bbox: crate::msg::BoundingBox2D::into_rmw_message(std::borrow::Cow::Owned(msg.bbox)).into_owned(),
        id: msg.id.as_str().into(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        results: msg.results
          .iter()
          .map(|elem| crate::msg::ObjectHypothesisWithPose::into_rmw_message(std::borrow::Cow::Borrowed(elem)).into_owned())
          .collect(),
        bbox: crate::msg::BoundingBox2D::into_rmw_message(std::borrow::Cow::Borrowed(&msg.bbox)).into_owned(),
        id: msg.id.as_str().into(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      results: msg.results
          .into_iter()
          .map(crate::msg::ObjectHypothesisWithPose::from_rmw_message)
          .collect(),
      bbox: crate::msg::BoundingBox2D::from_rmw_message(msg.bbox),
      id: msg.id.to_string(),
    }
  }
}


#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Detection3DArray {
    pub header: std_msgs::msg::Header,
    pub detections: Vec<crate::msg::Detection3D>,
}



impl Default for Detection3DArray {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(crate::msg::rmw::Detection3DArray::default())
  }
}

impl rosidl_runtime_rs::Message for Detection3DArray {
  type RmwMsg = crate::msg::rmw::Detection3DArray;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        detections: msg.detections
          .into_iter()
          .map(|elem| crate::msg::Detection3D::into_rmw_message(std::borrow::Cow::Owned(elem)).into_owned())
          .collect(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        detections: msg.detections
          .iter()
          .map(|elem| crate::msg::Detection3D::into_rmw_message(std::borrow::Cow::Borrowed(elem)).into_owned())
          .collect(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      detections: msg.detections
          .into_iter()
          .map(crate::msg::Detection3D::from_rmw_message)
          .collect(),
    }
  }
}


#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Detection3D {
    pub header: std_msgs::msg::Header,
    pub results: Vec<crate::msg::ObjectHypothesisWithPose>,
    pub bbox: crate::msg::BoundingBox3D,
    pub id: std::string::String,
}



impl Default for Detection3D {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(crate::msg::rmw::Detection3D::default())
  }
}

impl rosidl_runtime_rs::Message for Detection3D {
  type RmwMsg = crate::msg::rmw::Detection3D;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        results: msg.results
          .into_iter()
          .map(|elem| crate::msg::ObjectHypothesisWithPose::into_rmw_message(std::borrow::Cow::Owned(elem)).into_owned())
          .collect(),
        bbox: crate::msg::BoundingBox3D::into_rmw_message(std::borrow::Cow::Owned(msg.bbox)).into_owned(),
        id: msg.id.as_str().into(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        results: msg.results
          .iter()
          .map(|elem| crate::msg::ObjectHypothesisWithPose::into_rmw_message(std::borrow::Cow::Borrowed(elem)).into_owned())
          .collect(),
        bbox: crate::msg::BoundingBox3D::into_rmw_message(std::borrow::Cow::Borrowed(&msg.bbox)).into_owned(),
        id: msg.id.as_str().into(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      results: msg.results
          .into_iter()
          .map(crate::msg::ObjectHypothesisWithPose::from_rmw_message)
          .collect(),
      bbox: crate::msg::BoundingBox3D::from_rmw_message(msg.bbox),
      id: msg.id.to_string(),
    }
  }
}


#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct LabelInfo {
    pub header: std_msgs::msg::Header,
    pub class_map: Vec<crate::msg::VisionClass>,
    pub threshold: f32,
}



impl Default for LabelInfo {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(crate::msg::rmw::LabelInfo::default())
  }
}

impl rosidl_runtime_rs::Message for LabelInfo {
  type RmwMsg = crate::msg::rmw::LabelInfo;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        class_map: msg.class_map
          .into_iter()
          .map(|elem| crate::msg::VisionClass::into_rmw_message(std::borrow::Cow::Owned(elem)).into_owned())
          .collect(),
        threshold: msg.threshold,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        class_map: msg.class_map
          .iter()
          .map(|elem| crate::msg::VisionClass::into_rmw_message(std::borrow::Cow::Borrowed(elem)).into_owned())
          .collect(),
      threshold: msg.threshold,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      class_map: msg.class_map
          .into_iter()
          .map(crate::msg::VisionClass::from_rmw_message)
          .collect(),
      threshold: msg.threshold,
    }
  }
}


#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct ObjectHypothesis {
    pub class_id: std::string::String,
    pub score: f64,
}



impl Default for ObjectHypothesis {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(crate::msg::rmw::ObjectHypothesis::default())
  }
}

impl rosidl_runtime_rs::Message for ObjectHypothesis {
  type RmwMsg = crate::msg::rmw::ObjectHypothesis;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        class_id: msg.class_id.as_str().into(),
        score: msg.score,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        class_id: msg.class_id.as_str().into(),
      score: msg.score,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      class_id: msg.class_id.to_string(),
      score: msg.score,
    }
  }
}


#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct ObjectHypothesisWithPose {
    pub hypothesis: crate::msg::ObjectHypothesis,
    pub pose: geometry_msgs::msg::PoseWithCovariance,
}



impl Default for ObjectHypothesisWithPose {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(crate::msg::rmw::ObjectHypothesisWithPose::default())
  }
}

impl rosidl_runtime_rs::Message for ObjectHypothesisWithPose {
  type RmwMsg = crate::msg::rmw::ObjectHypothesisWithPose;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        hypothesis: crate::msg::ObjectHypothesis::into_rmw_message(std::borrow::Cow::Owned(msg.hypothesis)).into_owned(),
        pose: geometry_msgs::msg::PoseWithCovariance::into_rmw_message(std::borrow::Cow::Owned(msg.pose)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        hypothesis: crate::msg::ObjectHypothesis::into_rmw_message(std::borrow::Cow::Borrowed(&msg.hypothesis)).into_owned(),
        pose: geometry_msgs::msg::PoseWithCovariance::into_rmw_message(std::borrow::Cow::Borrowed(&msg.pose)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      hypothesis: crate::msg::ObjectHypothesis::from_rmw_message(msg.hypothesis),
      pose: geometry_msgs::msg::PoseWithCovariance::from_rmw_message(msg.pose),
    }
  }
}


#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct VisionClass {
    pub class_id: u16,
    pub class_name: std::string::String,
}



impl Default for VisionClass {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(crate::msg::rmw::VisionClass::default())
  }
}

impl rosidl_runtime_rs::Message for VisionClass {
  type RmwMsg = crate::msg::rmw::VisionClass;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        class_id: msg.class_id,
        class_name: msg.class_name.as_str().into(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      class_id: msg.class_id,
        class_name: msg.class_name.as_str().into(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      class_id: msg.class_id,
      class_name: msg.class_name.to_string(),
    }
  }
}


#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}



impl Default for Point2D {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(crate::msg::rmw::Point2D::default())
  }
}

impl rosidl_runtime_rs::Message for Point2D {
  type RmwMsg = crate::msg::rmw::Point2D;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        x: msg.x,
        y: msg.y,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      x: msg.x,
      y: msg.y,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      x: msg.x,
      y: msg.y,
    }
  }
}


#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Pose2D {
    pub position: crate::msg::Point2D,
    pub theta: f64,
}



impl Default for Pose2D {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(crate::msg::rmw::Pose2D::default())
  }
}

impl rosidl_runtime_rs::Message for Pose2D {
  type RmwMsg = crate::msg::rmw::Pose2D;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        position: crate::msg::Point2D::into_rmw_message(std::borrow::Cow::Owned(msg.position)).into_owned(),
        theta: msg.theta,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        position: crate::msg::Point2D::into_rmw_message(std::borrow::Cow::Borrowed(&msg.position)).into_owned(),
      theta: msg.theta,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      position: crate::msg::Point2D::from_rmw_message(msg.position),
      theta: msg.theta,
    }
  }
}


#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct VisionInfo {
    pub header: std_msgs::msg::Header,
    pub method: std::string::String,
    pub database_location: std::string::String,
    pub database_version: i32,
}



impl Default for VisionInfo {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(crate::msg::rmw::VisionInfo::default())
  }
}

impl rosidl_runtime_rs::Message for VisionInfo {
  type RmwMsg = crate::msg::rmw::VisionInfo;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        method: msg.method.as_str().into(),
        database_location: msg.database_location.as_str().into(),
        database_version: msg.database_version,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        method: msg.method.as_str().into(),
        database_location: msg.database_location.as_str().into(),
      database_version: msg.database_version,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      method: msg.method.to_string(),
      database_location: msg.database_location.to_string(),
      database_version: msg.database_version,
    }
  }
}


