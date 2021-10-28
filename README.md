# Arrow generated IPC format

The generated flatbuffers code for Rust.

Note that these files suffered modifications because flatbuffers is unable to compile Apache Arrow,
see https://github.com/google/flatbuffers/issues/6886.

* Google flatbuffers version 2.0
* Apache Arrow version 6.0

## Cargo features

This package is divided in 3 features:

* `ipc` for Apache [IPC format](https://github.com/apache/arrow/tree/master/format)
* `flight-data` for the `prost` part of the Apache [Flight protocol](https://github.com/apache/arrow/blob/master/format/Flight.proto)
* `flight-service` for the `tonic` part of the Apache [Flight protocol](https://github.com/apache/arrow/blob/master/format/Flight.proto)
