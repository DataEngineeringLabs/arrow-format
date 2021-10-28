//! Generated code of protobuf (Flight.proto) in [format](https://github.com/apache/arrow/tree/master/format)
// This file was automatically generated through the build.rs script, and should not be edited.

///
/// The request that a client provides to a server on handshake.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct HandshakeRequest {
    ///
    /// A defined protocol version
    #[prost(uint64, tag = "1")]
    pub protocol_version: u64,
    ///
    /// Arbitrary auth/handshake info.
    #[prost(bytes = "vec", tag = "2")]
    pub payload: ::prost::alloc::vec::Vec<u8>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct HandshakeResponse {
    ///
    /// A defined protocol version
    #[prost(uint64, tag = "1")]
    pub protocol_version: u64,
    ///
    /// Arbitrary auth/handshake info.
    #[prost(bytes = "vec", tag = "2")]
    pub payload: ::prost::alloc::vec::Vec<u8>,
}
///
/// A message for doing simple auth.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BasicAuth {
    #[prost(string, tag = "2")]
    pub username: ::prost::alloc::string::String,
    #[prost(string, tag = "3")]
    pub password: ::prost::alloc::string::String,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Empty {}
///
/// Describes an available action, including both the name used for execution
/// along with a short description of the purpose of the action.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ActionType {
    #[prost(string, tag = "1")]
    pub r#type: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub description: ::prost::alloc::string::String,
}
///
/// A service specific expression that can be used to return a limited set
/// of available Arrow Flight streams.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Criteria {
    #[prost(bytes = "vec", tag = "1")]
    pub expression: ::prost::alloc::vec::Vec<u8>,
}
///
/// An opaque action specific for the service.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Action {
    #[prost(string, tag = "1")]
    pub r#type: ::prost::alloc::string::String,
    #[prost(bytes = "vec", tag = "2")]
    pub body: ::prost::alloc::vec::Vec<u8>,
}
///
/// An opaque result returned after executing an action.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Result {
    #[prost(bytes = "vec", tag = "1")]
    pub body: ::prost::alloc::vec::Vec<u8>,
}
///
/// Wrap the result of a getSchema call
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SchemaResult {
    /// schema of the dataset as described in Schema.fbs::Schema.
    #[prost(bytes = "vec", tag = "1")]
    pub schema: ::prost::alloc::vec::Vec<u8>,
}
///
/// The name or tag for a Flight. May be used as a way to retrieve or generate
/// a flight or be used to expose a set of previously defined flights.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FlightDescriptor {
    #[prost(enumeration = "flight_descriptor::DescriptorType", tag = "1")]
    pub r#type: i32,
    ///
    /// Opaque value used to express a command. Should only be defined when
    /// type = CMD.
    #[prost(bytes = "vec", tag = "2")]
    pub cmd: ::prost::alloc::vec::Vec<u8>,
    ///
    /// List of strings identifying a particular dataset. Should only be defined
    /// when type = PATH.
    #[prost(string, repeated, tag = "3")]
    pub path: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}
/// Nested message and enum types in `FlightDescriptor`.
pub mod flight_descriptor {
    ///
    /// Describes what type of descriptor is defined.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum DescriptorType {
        /// Protobuf pattern, not used.
        Unknown = 0,
        ///
        /// A named path that identifies a dataset. A path is composed of a string
        /// or list of strings describing a particular dataset. This is conceptually
        ///  similar to a path inside a filesystem.
        Path = 1,
        ///
        /// An opaque command to generate a dataset.
        Cmd = 2,
    }
}
///
/// The access coordinates for retrieval of a dataset. With a FlightInfo, a
/// consumer is able to determine how to retrieve a dataset.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FlightInfo {
    /// schema of the dataset as described in Schema.fbs::Schema.
    #[prost(bytes = "vec", tag = "1")]
    pub schema: ::prost::alloc::vec::Vec<u8>,
    ///
    /// The descriptor associated with this info.
    #[prost(message, optional, tag = "2")]
    pub flight_descriptor: ::core::option::Option<FlightDescriptor>,
    ///
    /// A list of endpoints associated with the flight. To consume the whole
    /// flight, all endpoints must be consumed.
    #[prost(message, repeated, tag = "3")]
    pub endpoint: ::prost::alloc::vec::Vec<FlightEndpoint>,
    /// Set these to -1 if unknown.
    #[prost(int64, tag = "4")]
    pub total_records: i64,
    #[prost(int64, tag = "5")]
    pub total_bytes: i64,
}
///
/// A particular stream or split associated with a flight.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FlightEndpoint {
    ///
    /// Token used to retrieve this stream.
    #[prost(message, optional, tag = "1")]
    pub ticket: ::core::option::Option<Ticket>,
    ///
    /// A list of URIs where this ticket can be redeemed. If the list is
    /// empty, the expectation is that the ticket can only be redeemed on the
    /// current service where the ticket was generated.
    #[prost(message, repeated, tag = "2")]
    pub location: ::prost::alloc::vec::Vec<Location>,
}
///
/// A location where a Flight service will accept retrieval of a particular
/// stream given a ticket.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Location {
    #[prost(string, tag = "1")]
    pub uri: ::prost::alloc::string::String,
}
///
/// An opaque identifier that the service can use to retrieve a particular
/// portion of a stream.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Ticket {
    #[prost(bytes = "vec", tag = "1")]
    pub ticket: ::prost::alloc::vec::Vec<u8>,
}
///
/// A batch of Arrow data as part of a stream of batches.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FlightData {
    ///
    /// The descriptor of the data. This is only relevant when a client is
    /// starting a new DoPut stream.
    #[prost(message, optional, tag = "1")]
    pub flight_descriptor: ::core::option::Option<FlightDescriptor>,
    ///
    /// Header for message data as described in Message.fbs::Message.
    #[prost(bytes = "vec", tag = "2")]
    pub data_header: ::prost::alloc::vec::Vec<u8>,
    ///
    /// Application-defined metadata.
    #[prost(bytes = "vec", tag = "3")]
    pub app_metadata: ::prost::alloc::vec::Vec<u8>,
    ///
    /// The actual batch of Arrow data. Preferably handled with minimal-copies
    /// coming last in the definition to help with sidecar patterns (it is
    /// expected that some implementations will fetch this field off the wire
    /// with specialized code to avoid extra memory copies).
    #[prost(bytes = "vec", tag = "1000")]
    pub data_body: ::prost::alloc::vec::Vec<u8>,
}
///*
/// The response message associated with the submission of a DoPut.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PutResult {
    #[prost(bytes = "vec", tag = "1")]
    pub app_metadata: ::prost::alloc::vec::Vec<u8>,
}
