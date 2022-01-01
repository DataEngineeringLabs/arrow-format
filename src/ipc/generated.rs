pub mod org {
    pub mod apache {
        pub mod arrow {
            pub mod flatbuf {
                use std::convert::TryFrom;
                use std::convert::TryInto;

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct TensorDim {
                    pub size: i64,
                    pub name: Option<String>,
                }

                impl TensorDim {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(
                        builder: &mut planus::Builder,
                        size: impl planus::WriteAsDefault<i64, i64>,
                        name: impl planus::WriteAsOptional<planus::Offset<str>>,
                    ) -> planus::Offset<Self> {
                        let prepared_size = size.prepare(builder, &0);

                        let prepared_name = name.prepare(builder);

                        let mut table_writer =
                            planus::table_writer::TableWriter::<6, 12>::new(builder);

                        if prepared_size.is_some() {
                            table_writer.calculate_size::<i64>(2);
                        }
                        if prepared_name.is_some() {
                            table_writer.calculate_size::<planus::Offset<str>>(4);
                        }

                        table_writer.finish_calculating();

                        unsafe {
                            if let Some(prepared_size) = prepared_size {
                                table_writer.write::<_, _, 8>(0, &prepared_size);
                            }
                            if let Some(prepared_name) = prepared_name {
                                table_writer.write::<_, _, 4>(1, &prepared_name);
                            }
                        }

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<TensorDim>> for TensorDim {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<TensorDim> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<TensorDim>> for TensorDim {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<TensorDim>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<TensorDim> for TensorDim {
                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<TensorDim> {
                        TensorDim::create(builder, &self.size, &self.name)
                    }
                }

                #[derive(Copy, Clone)]
                pub struct TensorDimRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> TensorDimRef<'a> {
                    pub fn size(&self) -> planus::Result<i64> {
                        Ok(self.0.access(0, "TensorDim", "size")?.unwrap_or(0))
                    }

                    pub fn name(&self) -> planus::Result<Option<&'a str>> {
                        self.0.access(1, "TensorDim", "name")
                    }
                }

                impl<'a> std::fmt::Debug for TensorDimRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("TensorDimRef");
                        if let Ok(size) = self.size() {
                            f.field("size", &size);
                        }
                        if let Ok(Some(name)) = self.name() {
                            f.field("name", &name);
                        }
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for TensorDimRef<'a> {
                    type Value = TensorDim;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(TensorDim {
                            size: planus::ToOwned::to_owned(self.size()?)?,
                            name: if let Some(name) = self.name()? {
                                Some(planus::ToOwned::to_owned(name)?)
                            } else {
                                None
                            },
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for TensorDimRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for TensorDim {
                    type Output = planus::Result<TensorDimRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[TensorDimRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<TensorDim>> for TensorDim {
                    type Value = planus::Offset<TensorDim>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<TensorDim>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for TensorDimRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location("[TensorDimRef]", "read_as_root", 0)
                        })
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct Tensor {
                    pub type_: self::Type,
                    pub shape: Vec<self::TensorDim>,
                    pub strides: Option<Vec<i64>>,
                    pub data: self::Buffer,
                }

                impl Tensor {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(
                        builder: &mut planus::Builder,
                        type_: impl planus::WriteAsUnion<self::Type>,
                        shape: impl planus::WriteAs<planus::Offset<[planus::Offset<self::TensorDim>]>>,
                        strides: impl planus::WriteAsOptional<planus::Offset<[i64]>>,
                        data: impl planus::WriteAs<self::Buffer>,
                    ) -> planus::Offset<Self> {
                        let prepared_type_ = type_.prepare(builder);

                        let prepared_shape = shape.prepare(builder);

                        let prepared_strides = strides.prepare(builder);

                        let prepared_data = data.prepare(builder);

                        let mut table_writer =
                            planus::table_writer::TableWriter::<12, 29>::new(builder);

                        table_writer.calculate_size::<u8>(2);
                        table_writer.calculate_size::<planus::Offset<self::Type>>(4);
                        table_writer
                            .calculate_size::<planus::Offset<[planus::Offset<self::TensorDim>]>>(6);
                        if prepared_strides.is_some() {
                            table_writer.calculate_size::<planus::Offset<[i64]>>(8);
                        }
                        table_writer.calculate_size::<self::Buffer>(10);

                        table_writer.finish_calculating();

                        unsafe {
                            table_writer.write::<_, _, 16>(4, &prepared_data);
                            table_writer.write::<_, _, 4>(1, &prepared_type_.offset);
                            table_writer.write::<_, _, 4>(2, &prepared_shape);
                            if let Some(prepared_strides) = prepared_strides {
                                table_writer.write::<_, _, 4>(3, &prepared_strides);
                            }
                            table_writer.write::<_, _, 1>(0, &prepared_type_.tag);
                        }

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<Tensor>> for Tensor {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Tensor> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<Tensor>> for Tensor {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<Tensor>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<Tensor> for Tensor {
                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Tensor> {
                        Tensor::create(builder, &self.type_, &self.shape, &self.strides, &self.data)
                    }
                }

                #[derive(Copy, Clone)]
                pub struct TensorRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> TensorRef<'a> {
                    pub fn type_(&self) -> planus::Result<self::TypeRef<'a>> {
                        self.0.access_union_required(0, "Tensor", "type_")
                    }

                    pub fn shape(&self) -> planus::Result<planus::Vector<'a, self::TensorDim>> {
                        self.0.access_required(2, "Tensor", "shape")
                    }

                    pub fn strides(&self) -> planus::Result<Option<planus::Vector<'a, i64>>> {
                        self.0.access(3, "Tensor", "strides")
                    }

                    pub fn data(&self) -> planus::Result<self::BufferRef<'a>> {
                        self.0.access_required(4, "Tensor", "data")
                    }
                }

                impl<'a> std::fmt::Debug for TensorRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("TensorRef");
                        if let Ok(type_) = self.type_() {
                            f.field("type_", &type_);
                        }
                        if let Ok(shape) = self.shape() {
                            f.field("shape", &shape);
                        }
                        if let Ok(Some(strides)) = self.strides() {
                            f.field("strides", &strides);
                        }
                        if let Ok(data) = self.data() {
                            f.field("data", &data);
                        }
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for TensorRef<'a> {
                    type Value = Tensor;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(Tensor {
                            type_: planus::ToOwned::to_owned(self.type_()?)?,
                            shape: planus::ToOwned::to_owned(self.shape()?)?,
                            strides: if let Some(strides) = self.strides()? {
                                Some(planus::ToOwned::to_owned(strides)?)
                            } else {
                                None
                            },
                            data: planus::ToOwned::to_owned(self.data()?)?,
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for TensorRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for Tensor {
                    type Output = planus::Result<TensorRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[TensorRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<Tensor>> for Tensor {
                    type Value = planus::Offset<Tensor>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<Tensor>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for TensorRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location("[TensorRef]", "read_as_root", 0)
                        })
                    }
                }

                #[derive(
                    Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize,
                )]
                #[repr(i16)]
                pub enum MetadataVersion {
                    V1 = 0,
                    V2 = 1,
                    V3 = 2,
                    V4 = 3,
                    V5 = 4,
                }

                impl TryFrom<i16> for MetadataVersion {
                    type Error = planus::errors::UnknownEnumTagKind;
                    fn try_from(
                        value: i16,
                    ) -> std::result::Result<Self, planus::errors::UnknownEnumTagKind>
                    {
                        #[allow(clippy::match_single_binding)]
                        match value {
                            0 => Ok(MetadataVersion::V1),
                            1 => Ok(MetadataVersion::V2),
                            2 => Ok(MetadataVersion::V3),
                            3 => Ok(MetadataVersion::V4),
                            4 => Ok(MetadataVersion::V5),

                            _ => Err(planus::errors::UnknownEnumTagKind { tag: value as i128 }),
                        }
                    }
                }

                impl From<MetadataVersion> for i16 {
                    fn from(value: MetadataVersion) -> Self {
                        value as i16
                    }
                }

                impl planus::ToOwned for MetadataVersion {
                    type Value = MetadataVersion;
                    #[inline]
                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(self)
                    }
                }

                impl planus::Primitive for MetadataVersion {
                    const ALIGNMENT: usize = 2;
                    const SIZE: usize = 2;
                }

                impl planus::WriteAsPrimitive<MetadataVersion> for MetadataVersion {
                    #[inline]
                    fn write<const N: usize>(
                        &self,
                        cursor: planus::Cursor<'_, N>,
                        buffer_position: u32,
                    ) {
                        (*self as i16).write(cursor, buffer_position);
                    }
                }

                impl planus::WriteAs<MetadataVersion> for MetadataVersion {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(&self, _builder: &mut planus::Builder) -> MetadataVersion {
                        *self
                    }
                }

                impl planus::WriteAsDefault<MetadataVersion, MetadataVersion> for MetadataVersion {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(
                        &self,
                        _builder: &mut planus::Builder,
                        default: &MetadataVersion,
                    ) -> Option<MetadataVersion> {
                        if self == default {
                            None
                        } else {
                            Some(*self)
                        }
                    }
                }

                impl planus::WriteAsOptional<MetadataVersion> for MetadataVersion {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(&self, _builder: &mut planus::Builder) -> Option<MetadataVersion> {
                        Some(*self)
                    }
                }

                impl<'buf> planus::TableRead<'buf> for MetadataVersion {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'buf>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        let n: i16 = planus::TableRead::from_buffer(buffer, offset)?;
                        Ok(n.try_into()?)
                    }
                }

                impl<'buf> planus::VectorRead<'buf> for MetadataVersion {
                    type Output = std::result::Result<Self, planus::errors::UnknownEnumTag>;

                    const STRIDE: usize = 2;
                    #[inline]
                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'buf>,
                        offset: usize,
                    ) -> Self::Output {
                        let value = <i16 as planus::VectorRead>::from_buffer(buffer, offset);
                        let value: std::result::Result<Self, _> = value.try_into();
                        value.map_err(|error_kind| {
                            error_kind.with_error_location(
                                "MetadataVersion",
                                "VectorRead::from_buffer",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl<'buf> planus::VectorWrite<MetadataVersion> for MetadataVersion {
                    const STRIDE: usize = 2;

                    type Value = Self;

                    fn prepare(&self, _builder: &mut planus::Builder) -> Self::Value {
                        *self
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[Self],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 2];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (2 * i) as u32,
                            );
                        }
                    }
                }

                #[derive(
                    Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize,
                )]
                #[repr(i64)]
                pub enum Feature {
                    Unused = 0,
                    DictionaryReplacement = 1,
                    CompressedBody = 2,
                }

                impl TryFrom<i64> for Feature {
                    type Error = planus::errors::UnknownEnumTagKind;
                    fn try_from(
                        value: i64,
                    ) -> std::result::Result<Self, planus::errors::UnknownEnumTagKind>
                    {
                        #[allow(clippy::match_single_binding)]
                        match value {
                            0 => Ok(Feature::Unused),
                            1 => Ok(Feature::DictionaryReplacement),
                            2 => Ok(Feature::CompressedBody),

                            _ => Err(planus::errors::UnknownEnumTagKind { tag: value as i128 }),
                        }
                    }
                }

                impl From<Feature> for i64 {
                    fn from(value: Feature) -> Self {
                        value as i64
                    }
                }

                impl planus::ToOwned for Feature {
                    type Value = Feature;
                    #[inline]
                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(self)
                    }
                }

                impl planus::Primitive for Feature {
                    const ALIGNMENT: usize = 8;
                    const SIZE: usize = 8;
                }

                impl planus::WriteAsPrimitive<Feature> for Feature {
                    #[inline]
                    fn write<const N: usize>(
                        &self,
                        cursor: planus::Cursor<'_, N>,
                        buffer_position: u32,
                    ) {
                        (*self as i64).write(cursor, buffer_position);
                    }
                }

                impl planus::WriteAs<Feature> for Feature {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(&self, _builder: &mut planus::Builder) -> Feature {
                        *self
                    }
                }

                impl planus::WriteAsDefault<Feature, Feature> for Feature {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(
                        &self,
                        _builder: &mut planus::Builder,
                        default: &Feature,
                    ) -> Option<Feature> {
                        if self == default {
                            None
                        } else {
                            Some(*self)
                        }
                    }
                }

                impl planus::WriteAsOptional<Feature> for Feature {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(&self, _builder: &mut planus::Builder) -> Option<Feature> {
                        Some(*self)
                    }
                }

                impl<'buf> planus::TableRead<'buf> for Feature {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'buf>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        let n: i64 = planus::TableRead::from_buffer(buffer, offset)?;
                        Ok(n.try_into()?)
                    }
                }

                impl<'buf> planus::VectorRead<'buf> for Feature {
                    type Output = std::result::Result<Self, planus::errors::UnknownEnumTag>;

                    const STRIDE: usize = 8;
                    #[inline]
                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'buf>,
                        offset: usize,
                    ) -> Self::Output {
                        let value = <i64 as planus::VectorRead>::from_buffer(buffer, offset);
                        let value: std::result::Result<Self, _> = value.try_into();
                        value.map_err(|error_kind| {
                            error_kind.with_error_location(
                                "Feature",
                                "VectorRead::from_buffer",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl<'buf> planus::VectorWrite<Feature> for Feature {
                    const STRIDE: usize = 8;

                    type Value = Self;

                    fn prepare(&self, _builder: &mut planus::Builder) -> Self::Value {
                        *self
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[Self],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 8];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (8 * i) as u32,
                            );
                        }
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct Null {}

                impl Null {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(builder: &mut planus::Builder) -> planus::Offset<Self> {
                        let mut table_writer =
                            planus::table_writer::TableWriter::<4, 0>::new(builder);

                        table_writer.finish_calculating();

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<Null>> for Null {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Null> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<Null>> for Null {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<Null>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<Null> for Null {
                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Null> {
                        Null::create(builder)
                    }
                }

                #[derive(Copy, Clone)]
                pub struct NullRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> NullRef<'a> {}

                impl<'a> std::fmt::Debug for NullRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("NullRef");

                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for NullRef<'a> {
                    type Value = Null;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(Null {})
                    }
                }

                impl<'a> planus::TableRead<'a> for NullRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for Null {
                    type Output = planus::Result<NullRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[NullRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<Null>> for Null {
                    type Value = planus::Offset<Null>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<Null>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for NullRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location("[NullRef]", "read_as_root", 0)
                        })
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct Struct {}

                impl Struct {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(builder: &mut planus::Builder) -> planus::Offset<Self> {
                        let mut table_writer =
                            planus::table_writer::TableWriter::<4, 0>::new(builder);

                        table_writer.finish_calculating();

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<Struct>> for Struct {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Struct> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<Struct>> for Struct {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<Struct>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<Struct> for Struct {
                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Struct> {
                        Struct::create(builder)
                    }
                }

                #[derive(Copy, Clone)]
                pub struct StructRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> StructRef<'a> {}

                impl<'a> std::fmt::Debug for StructRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("StructRef");

                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for StructRef<'a> {
                    type Value = Struct;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(Struct {})
                    }
                }

                impl<'a> planus::TableRead<'a> for StructRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for Struct {
                    type Output = planus::Result<StructRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[StructRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<Struct>> for Struct {
                    type Value = planus::Offset<Struct>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<Struct>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for StructRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location("[StructRef]", "read_as_root", 0)
                        })
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct List {}

                impl List {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(builder: &mut planus::Builder) -> planus::Offset<Self> {
                        let mut table_writer =
                            planus::table_writer::TableWriter::<4, 0>::new(builder);

                        table_writer.finish_calculating();

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<List>> for List {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<List> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<List>> for List {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<List>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<List> for List {
                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<List> {
                        List::create(builder)
                    }
                }

                #[derive(Copy, Clone)]
                pub struct ListRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> ListRef<'a> {}

                impl<'a> std::fmt::Debug for ListRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("ListRef");

                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for ListRef<'a> {
                    type Value = List;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(List {})
                    }
                }

                impl<'a> planus::TableRead<'a> for ListRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for List {
                    type Output = planus::Result<ListRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[ListRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<List>> for List {
                    type Value = planus::Offset<List>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<List>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for ListRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location("[ListRef]", "read_as_root", 0)
                        })
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct LargeList {}

                impl LargeList {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(builder: &mut planus::Builder) -> planus::Offset<Self> {
                        let mut table_writer =
                            planus::table_writer::TableWriter::<4, 0>::new(builder);

                        table_writer.finish_calculating();

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<LargeList>> for LargeList {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<LargeList> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<LargeList>> for LargeList {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<LargeList>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<LargeList> for LargeList {
                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<LargeList> {
                        LargeList::create(builder)
                    }
                }

                #[derive(Copy, Clone)]
                pub struct LargeListRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> LargeListRef<'a> {}

                impl<'a> std::fmt::Debug for LargeListRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("LargeListRef");

                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for LargeListRef<'a> {
                    type Value = LargeList;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(LargeList {})
                    }
                }

                impl<'a> planus::TableRead<'a> for LargeListRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for LargeList {
                    type Output = planus::Result<LargeListRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[LargeListRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<LargeList>> for LargeList {
                    type Value = planus::Offset<LargeList>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<LargeList>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for LargeListRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location("[LargeListRef]", "read_as_root", 0)
                        })
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct FixedSizeList {
                    pub list_size: i32,
                }

                impl FixedSizeList {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(
                        builder: &mut planus::Builder,
                        list_size: impl planus::WriteAsDefault<i32, i32>,
                    ) -> planus::Offset<Self> {
                        let prepared_list_size = list_size.prepare(builder, &0);

                        let mut table_writer =
                            planus::table_writer::TableWriter::<4, 4>::new(builder);

                        if prepared_list_size.is_some() {
                            table_writer.calculate_size::<i32>(2);
                        }

                        table_writer.finish_calculating();

                        unsafe {
                            if let Some(prepared_list_size) = prepared_list_size {
                                table_writer.write::<_, _, 4>(0, &prepared_list_size);
                            }
                        }

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<FixedSizeList>> for FixedSizeList {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> planus::Offset<FixedSizeList> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<FixedSizeList>> for FixedSizeList {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<FixedSizeList>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<FixedSizeList> for FixedSizeList {
                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> planus::Offset<FixedSizeList> {
                        FixedSizeList::create(builder, &self.list_size)
                    }
                }

                #[derive(Copy, Clone)]
                pub struct FixedSizeListRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> FixedSizeListRef<'a> {
                    pub fn list_size(&self) -> planus::Result<i32> {
                        Ok(self.0.access(0, "FixedSizeList", "list_size")?.unwrap_or(0))
                    }
                }

                impl<'a> std::fmt::Debug for FixedSizeListRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("FixedSizeListRef");
                        if let Ok(list_size) = self.list_size() {
                            f.field("list_size", &list_size);
                        }
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for FixedSizeListRef<'a> {
                    type Value = FixedSizeList;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(FixedSizeList {
                            list_size: planus::ToOwned::to_owned(self.list_size()?)?,
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for FixedSizeListRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for FixedSizeList {
                    type Output = planus::Result<FixedSizeListRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[FixedSizeListRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<FixedSizeList>> for FixedSizeList {
                    type Value = planus::Offset<FixedSizeList>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<FixedSizeList>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for FixedSizeListRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location("[FixedSizeListRef]", "read_as_root", 0)
                        })
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct Map {
                    pub keys_sorted: bool,
                }

                impl Map {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(
                        builder: &mut planus::Builder,
                        keys_sorted: impl planus::WriteAsDefault<bool, bool>,
                    ) -> planus::Offset<Self> {
                        let prepared_keys_sorted = keys_sorted.prepare(builder, &false);

                        let mut table_writer =
                            planus::table_writer::TableWriter::<4, 1>::new(builder);

                        if prepared_keys_sorted.is_some() {
                            table_writer.calculate_size::<bool>(2);
                        }

                        table_writer.finish_calculating();

                        unsafe {
                            if let Some(prepared_keys_sorted) = prepared_keys_sorted {
                                table_writer.write::<_, _, 1>(0, &prepared_keys_sorted);
                            }
                        }

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<Map>> for Map {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Map> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<Map>> for Map {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<Map>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<Map> for Map {
                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Map> {
                        Map::create(builder, &self.keys_sorted)
                    }
                }

                #[derive(Copy, Clone)]
                pub struct MapRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> MapRef<'a> {
                    pub fn keys_sorted(&self) -> planus::Result<bool> {
                        Ok(self.0.access(0, "Map", "keys_sorted")?.unwrap_or(false))
                    }
                }

                impl<'a> std::fmt::Debug for MapRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("MapRef");
                        if let Ok(keys_sorted) = self.keys_sorted() {
                            f.field("keys_sorted", &keys_sorted);
                        }
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for MapRef<'a> {
                    type Value = Map;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(Map {
                            keys_sorted: planus::ToOwned::to_owned(self.keys_sorted()?)?,
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for MapRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for Map {
                    type Output = planus::Result<MapRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[MapRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<Map>> for Map {
                    type Value = planus::Offset<Map>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<Map>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for MapRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location("[MapRef]", "read_as_root", 0)
                        })
                    }
                }

                #[derive(
                    Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize,
                )]
                #[repr(i16)]
                pub enum UnionMode {
                    Sparse = 0,
                    Dense = 1,
                }

                impl TryFrom<i16> for UnionMode {
                    type Error = planus::errors::UnknownEnumTagKind;
                    fn try_from(
                        value: i16,
                    ) -> std::result::Result<Self, planus::errors::UnknownEnumTagKind>
                    {
                        #[allow(clippy::match_single_binding)]
                        match value {
                            0 => Ok(UnionMode::Sparse),
                            1 => Ok(UnionMode::Dense),

                            _ => Err(planus::errors::UnknownEnumTagKind { tag: value as i128 }),
                        }
                    }
                }

                impl From<UnionMode> for i16 {
                    fn from(value: UnionMode) -> Self {
                        value as i16
                    }
                }

                impl planus::ToOwned for UnionMode {
                    type Value = UnionMode;
                    #[inline]
                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(self)
                    }
                }

                impl planus::Primitive for UnionMode {
                    const ALIGNMENT: usize = 2;
                    const SIZE: usize = 2;
                }

                impl planus::WriteAsPrimitive<UnionMode> for UnionMode {
                    #[inline]
                    fn write<const N: usize>(
                        &self,
                        cursor: planus::Cursor<'_, N>,
                        buffer_position: u32,
                    ) {
                        (*self as i16).write(cursor, buffer_position);
                    }
                }

                impl planus::WriteAs<UnionMode> for UnionMode {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(&self, _builder: &mut planus::Builder) -> UnionMode {
                        *self
                    }
                }

                impl planus::WriteAsDefault<UnionMode, UnionMode> for UnionMode {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(
                        &self,
                        _builder: &mut planus::Builder,
                        default: &UnionMode,
                    ) -> Option<UnionMode> {
                        if self == default {
                            None
                        } else {
                            Some(*self)
                        }
                    }
                }

                impl planus::WriteAsOptional<UnionMode> for UnionMode {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(&self, _builder: &mut planus::Builder) -> Option<UnionMode> {
                        Some(*self)
                    }
                }

                impl<'buf> planus::TableRead<'buf> for UnionMode {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'buf>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        let n: i16 = planus::TableRead::from_buffer(buffer, offset)?;
                        Ok(n.try_into()?)
                    }
                }

                impl<'buf> planus::VectorRead<'buf> for UnionMode {
                    type Output = std::result::Result<Self, planus::errors::UnknownEnumTag>;

                    const STRIDE: usize = 2;
                    #[inline]
                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'buf>,
                        offset: usize,
                    ) -> Self::Output {
                        let value = <i16 as planus::VectorRead>::from_buffer(buffer, offset);
                        let value: std::result::Result<Self, _> = value.try_into();
                        value.map_err(|error_kind| {
                            error_kind.with_error_location(
                                "UnionMode",
                                "VectorRead::from_buffer",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl<'buf> planus::VectorWrite<UnionMode> for UnionMode {
                    const STRIDE: usize = 2;

                    type Value = Self;

                    fn prepare(&self, _builder: &mut planus::Builder) -> Self::Value {
                        *self
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[Self],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 2];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (2 * i) as u32,
                            );
                        }
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct Union {
                    pub mode: self::UnionMode,
                    pub type_ids: Option<Vec<i32>>,
                }

                impl Union {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(
                        builder: &mut planus::Builder,
                        mode: impl planus::WriteAsDefault<self::UnionMode, self::UnionMode>,
                        type_ids: impl planus::WriteAsOptional<planus::Offset<[i32]>>,
                    ) -> planus::Offset<Self> {
                        let prepared_mode = mode.prepare(builder, &self::UnionMode::Sparse);

                        let prepared_type_ids = type_ids.prepare(builder);

                        let mut table_writer =
                            planus::table_writer::TableWriter::<6, 6>::new(builder);

                        if prepared_mode.is_some() {
                            table_writer.calculate_size::<self::UnionMode>(2);
                        }
                        if prepared_type_ids.is_some() {
                            table_writer.calculate_size::<planus::Offset<[i32]>>(4);
                        }

                        table_writer.finish_calculating();

                        unsafe {
                            if let Some(prepared_type_ids) = prepared_type_ids {
                                table_writer.write::<_, _, 4>(1, &prepared_type_ids);
                            }
                            if let Some(prepared_mode) = prepared_mode {
                                table_writer.write::<_, _, 2>(0, &prepared_mode);
                            }
                        }

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<Union>> for Union {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Union> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<Union>> for Union {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<Union>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<Union> for Union {
                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Union> {
                        Union::create(builder, &self.mode, &self.type_ids)
                    }
                }

                #[derive(Copy, Clone)]
                pub struct UnionRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> UnionRef<'a> {
                    pub fn mode(&self) -> planus::Result<self::UnionMode> {
                        Ok(self
                            .0
                            .access(0, "Union", "mode")?
                            .unwrap_or(self::UnionMode::Sparse))
                    }

                    pub fn type_ids(&self) -> planus::Result<Option<planus::Vector<'a, i32>>> {
                        self.0.access(1, "Union", "type_ids")
                    }
                }

                impl<'a> std::fmt::Debug for UnionRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("UnionRef");
                        if let Ok(mode) = self.mode() {
                            f.field("mode", &mode);
                        }
                        if let Ok(Some(type_ids)) = self.type_ids() {
                            f.field("type_ids", &type_ids);
                        }
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for UnionRef<'a> {
                    type Value = Union;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(Union {
                            mode: planus::ToOwned::to_owned(self.mode()?)?,
                            type_ids: if let Some(type_ids) = self.type_ids()? {
                                Some(planus::ToOwned::to_owned(type_ids)?)
                            } else {
                                None
                            },
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for UnionRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for Union {
                    type Output = planus::Result<UnionRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[UnionRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<Union>> for Union {
                    type Value = planus::Offset<Union>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<Union>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for UnionRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location("[UnionRef]", "read_as_root", 0)
                        })
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct Int {
                    pub bit_width: i32,
                    pub is_signed: bool,
                }

                impl Int {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(
                        builder: &mut planus::Builder,
                        bit_width: impl planus::WriteAsDefault<i32, i32>,
                        is_signed: impl planus::WriteAsDefault<bool, bool>,
                    ) -> planus::Offset<Self> {
                        let prepared_bit_width = bit_width.prepare(builder, &0);

                        let prepared_is_signed = is_signed.prepare(builder, &false);

                        let mut table_writer =
                            planus::table_writer::TableWriter::<6, 5>::new(builder);

                        if prepared_bit_width.is_some() {
                            table_writer.calculate_size::<i32>(2);
                        }
                        if prepared_is_signed.is_some() {
                            table_writer.calculate_size::<bool>(4);
                        }

                        table_writer.finish_calculating();

                        unsafe {
                            if let Some(prepared_bit_width) = prepared_bit_width {
                                table_writer.write::<_, _, 4>(0, &prepared_bit_width);
                            }
                            if let Some(prepared_is_signed) = prepared_is_signed {
                                table_writer.write::<_, _, 1>(1, &prepared_is_signed);
                            }
                        }

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<Int>> for Int {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Int> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<Int>> for Int {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<Int>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<Int> for Int {
                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Int> {
                        Int::create(builder, &self.bit_width, &self.is_signed)
                    }
                }

                #[derive(Copy, Clone)]
                pub struct IntRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> IntRef<'a> {
                    pub fn bit_width(&self) -> planus::Result<i32> {
                        Ok(self.0.access(0, "Int", "bit_width")?.unwrap_or(0))
                    }

                    pub fn is_signed(&self) -> planus::Result<bool> {
                        Ok(self.0.access(1, "Int", "is_signed")?.unwrap_or(false))
                    }
                }

                impl<'a> std::fmt::Debug for IntRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("IntRef");
                        if let Ok(bit_width) = self.bit_width() {
                            f.field("bit_width", &bit_width);
                        }
                        if let Ok(is_signed) = self.is_signed() {
                            f.field("is_signed", &is_signed);
                        }
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for IntRef<'a> {
                    type Value = Int;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(Int {
                            bit_width: planus::ToOwned::to_owned(self.bit_width()?)?,
                            is_signed: planus::ToOwned::to_owned(self.is_signed()?)?,
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for IntRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for Int {
                    type Output = planus::Result<IntRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[IntRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<Int>> for Int {
                    type Value = planus::Offset<Int>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<Int>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for IntRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location("[IntRef]", "read_as_root", 0)
                        })
                    }
                }

                #[derive(
                    Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize,
                )]
                #[repr(i16)]
                pub enum Precision {
                    Half = 0,
                    Single = 1,
                    Double = 2,
                }

                impl TryFrom<i16> for Precision {
                    type Error = planus::errors::UnknownEnumTagKind;
                    fn try_from(
                        value: i16,
                    ) -> std::result::Result<Self, planus::errors::UnknownEnumTagKind>
                    {
                        #[allow(clippy::match_single_binding)]
                        match value {
                            0 => Ok(Precision::Half),
                            1 => Ok(Precision::Single),
                            2 => Ok(Precision::Double),

                            _ => Err(planus::errors::UnknownEnumTagKind { tag: value as i128 }),
                        }
                    }
                }

                impl From<Precision> for i16 {
                    fn from(value: Precision) -> Self {
                        value as i16
                    }
                }

                impl planus::ToOwned for Precision {
                    type Value = Precision;
                    #[inline]
                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(self)
                    }
                }

                impl planus::Primitive for Precision {
                    const ALIGNMENT: usize = 2;
                    const SIZE: usize = 2;
                }

                impl planus::WriteAsPrimitive<Precision> for Precision {
                    #[inline]
                    fn write<const N: usize>(
                        &self,
                        cursor: planus::Cursor<'_, N>,
                        buffer_position: u32,
                    ) {
                        (*self as i16).write(cursor, buffer_position);
                    }
                }

                impl planus::WriteAs<Precision> for Precision {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(&self, _builder: &mut planus::Builder) -> Precision {
                        *self
                    }
                }

                impl planus::WriteAsDefault<Precision, Precision> for Precision {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(
                        &self,
                        _builder: &mut planus::Builder,
                        default: &Precision,
                    ) -> Option<Precision> {
                        if self == default {
                            None
                        } else {
                            Some(*self)
                        }
                    }
                }

                impl planus::WriteAsOptional<Precision> for Precision {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(&self, _builder: &mut planus::Builder) -> Option<Precision> {
                        Some(*self)
                    }
                }

                impl<'buf> planus::TableRead<'buf> for Precision {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'buf>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        let n: i16 = planus::TableRead::from_buffer(buffer, offset)?;
                        Ok(n.try_into()?)
                    }
                }

                impl<'buf> planus::VectorRead<'buf> for Precision {
                    type Output = std::result::Result<Self, planus::errors::UnknownEnumTag>;

                    const STRIDE: usize = 2;
                    #[inline]
                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'buf>,
                        offset: usize,
                    ) -> Self::Output {
                        let value = <i16 as planus::VectorRead>::from_buffer(buffer, offset);
                        let value: std::result::Result<Self, _> = value.try_into();
                        value.map_err(|error_kind| {
                            error_kind.with_error_location(
                                "Precision",
                                "VectorRead::from_buffer",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl<'buf> planus::VectorWrite<Precision> for Precision {
                    const STRIDE: usize = 2;

                    type Value = Self;

                    fn prepare(&self, _builder: &mut planus::Builder) -> Self::Value {
                        *self
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[Self],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 2];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (2 * i) as u32,
                            );
                        }
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct FloatingPoint {
                    pub precision: self::Precision,
                }

                impl FloatingPoint {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(
                        builder: &mut planus::Builder,
                        precision: impl planus::WriteAsDefault<self::Precision, self::Precision>,
                    ) -> planus::Offset<Self> {
                        let prepared_precision = precision.prepare(builder, &self::Precision::Half);

                        let mut table_writer =
                            planus::table_writer::TableWriter::<4, 2>::new(builder);

                        if prepared_precision.is_some() {
                            table_writer.calculate_size::<self::Precision>(2);
                        }

                        table_writer.finish_calculating();

                        unsafe {
                            if let Some(prepared_precision) = prepared_precision {
                                table_writer.write::<_, _, 2>(0, &prepared_precision);
                            }
                        }

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<FloatingPoint>> for FloatingPoint {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> planus::Offset<FloatingPoint> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<FloatingPoint>> for FloatingPoint {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<FloatingPoint>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<FloatingPoint> for FloatingPoint {
                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> planus::Offset<FloatingPoint> {
                        FloatingPoint::create(builder, &self.precision)
                    }
                }

                #[derive(Copy, Clone)]
                pub struct FloatingPointRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> FloatingPointRef<'a> {
                    pub fn precision(&self) -> planus::Result<self::Precision> {
                        Ok(self
                            .0
                            .access(0, "FloatingPoint", "precision")?
                            .unwrap_or(self::Precision::Half))
                    }
                }

                impl<'a> std::fmt::Debug for FloatingPointRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("FloatingPointRef");
                        if let Ok(precision) = self.precision() {
                            f.field("precision", &precision);
                        }
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for FloatingPointRef<'a> {
                    type Value = FloatingPoint;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(FloatingPoint {
                            precision: planus::ToOwned::to_owned(self.precision()?)?,
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for FloatingPointRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for FloatingPoint {
                    type Output = planus::Result<FloatingPointRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[FloatingPointRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<FloatingPoint>> for FloatingPoint {
                    type Value = planus::Offset<FloatingPoint>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<FloatingPoint>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for FloatingPointRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location("[FloatingPointRef]", "read_as_root", 0)
                        })
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct Utf8 {}

                impl Utf8 {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(builder: &mut planus::Builder) -> planus::Offset<Self> {
                        let mut table_writer =
                            planus::table_writer::TableWriter::<4, 0>::new(builder);

                        table_writer.finish_calculating();

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<Utf8>> for Utf8 {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Utf8> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<Utf8>> for Utf8 {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<Utf8>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<Utf8> for Utf8 {
                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Utf8> {
                        Utf8::create(builder)
                    }
                }

                #[derive(Copy, Clone)]
                pub struct Utf8Ref<'a>(planus::table_reader::Table<'a>);

                impl<'a> Utf8Ref<'a> {}

                impl<'a> std::fmt::Debug for Utf8Ref<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("Utf8Ref");

                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for Utf8Ref<'a> {
                    type Value = Utf8;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(Utf8 {})
                    }
                }

                impl<'a> planus::TableRead<'a> for Utf8Ref<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for Utf8 {
                    type Output = planus::Result<Utf8Ref<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[Utf8Ref]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<Utf8>> for Utf8 {
                    type Value = planus::Offset<Utf8>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<Utf8>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for Utf8Ref<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location("[Utf8Ref]", "read_as_root", 0)
                        })
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct Binary {}

                impl Binary {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(builder: &mut planus::Builder) -> planus::Offset<Self> {
                        let mut table_writer =
                            planus::table_writer::TableWriter::<4, 0>::new(builder);

                        table_writer.finish_calculating();

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<Binary>> for Binary {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Binary> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<Binary>> for Binary {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<Binary>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<Binary> for Binary {
                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Binary> {
                        Binary::create(builder)
                    }
                }

                #[derive(Copy, Clone)]
                pub struct BinaryRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> BinaryRef<'a> {}

                impl<'a> std::fmt::Debug for BinaryRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("BinaryRef");

                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for BinaryRef<'a> {
                    type Value = Binary;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(Binary {})
                    }
                }

                impl<'a> planus::TableRead<'a> for BinaryRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for Binary {
                    type Output = planus::Result<BinaryRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[BinaryRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<Binary>> for Binary {
                    type Value = planus::Offset<Binary>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<Binary>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for BinaryRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location("[BinaryRef]", "read_as_root", 0)
                        })
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct LargeUtf8 {}

                impl LargeUtf8 {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(builder: &mut planus::Builder) -> planus::Offset<Self> {
                        let mut table_writer =
                            planus::table_writer::TableWriter::<4, 0>::new(builder);

                        table_writer.finish_calculating();

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<LargeUtf8>> for LargeUtf8 {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<LargeUtf8> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<LargeUtf8>> for LargeUtf8 {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<LargeUtf8>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<LargeUtf8> for LargeUtf8 {
                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<LargeUtf8> {
                        LargeUtf8::create(builder)
                    }
                }

                #[derive(Copy, Clone)]
                pub struct LargeUtf8Ref<'a>(planus::table_reader::Table<'a>);

                impl<'a> LargeUtf8Ref<'a> {}

                impl<'a> std::fmt::Debug for LargeUtf8Ref<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("LargeUtf8Ref");

                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for LargeUtf8Ref<'a> {
                    type Value = LargeUtf8;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(LargeUtf8 {})
                    }
                }

                impl<'a> planus::TableRead<'a> for LargeUtf8Ref<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for LargeUtf8 {
                    type Output = planus::Result<LargeUtf8Ref<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[LargeUtf8Ref]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<LargeUtf8>> for LargeUtf8 {
                    type Value = planus::Offset<LargeUtf8>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<LargeUtf8>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for LargeUtf8Ref<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location("[LargeUtf8Ref]", "read_as_root", 0)
                        })
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct LargeBinary {}

                impl LargeBinary {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(builder: &mut planus::Builder) -> planus::Offset<Self> {
                        let mut table_writer =
                            planus::table_writer::TableWriter::<4, 0>::new(builder);

                        table_writer.finish_calculating();

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<LargeBinary>> for LargeBinary {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> planus::Offset<LargeBinary> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<LargeBinary>> for LargeBinary {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<LargeBinary>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<LargeBinary> for LargeBinary {
                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> planus::Offset<LargeBinary> {
                        LargeBinary::create(builder)
                    }
                }

                #[derive(Copy, Clone)]
                pub struct LargeBinaryRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> LargeBinaryRef<'a> {}

                impl<'a> std::fmt::Debug for LargeBinaryRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("LargeBinaryRef");

                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for LargeBinaryRef<'a> {
                    type Value = LargeBinary;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(LargeBinary {})
                    }
                }

                impl<'a> planus::TableRead<'a> for LargeBinaryRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for LargeBinary {
                    type Output = planus::Result<LargeBinaryRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[LargeBinaryRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<LargeBinary>> for LargeBinary {
                    type Value = planus::Offset<LargeBinary>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<LargeBinary>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for LargeBinaryRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location("[LargeBinaryRef]", "read_as_root", 0)
                        })
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct FixedSizeBinary {
                    pub byte_width: i32,
                }

                impl FixedSizeBinary {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(
                        builder: &mut planus::Builder,
                        byte_width: impl planus::WriteAsDefault<i32, i32>,
                    ) -> planus::Offset<Self> {
                        let prepared_byte_width = byte_width.prepare(builder, &0);

                        let mut table_writer =
                            planus::table_writer::TableWriter::<4, 4>::new(builder);

                        if prepared_byte_width.is_some() {
                            table_writer.calculate_size::<i32>(2);
                        }

                        table_writer.finish_calculating();

                        unsafe {
                            if let Some(prepared_byte_width) = prepared_byte_width {
                                table_writer.write::<_, _, 4>(0, &prepared_byte_width);
                            }
                        }

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<FixedSizeBinary>> for FixedSizeBinary {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> planus::Offset<FixedSizeBinary> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<FixedSizeBinary>> for FixedSizeBinary {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<FixedSizeBinary>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<FixedSizeBinary> for FixedSizeBinary {
                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> planus::Offset<FixedSizeBinary> {
                        FixedSizeBinary::create(builder, &self.byte_width)
                    }
                }

                #[derive(Copy, Clone)]
                pub struct FixedSizeBinaryRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> FixedSizeBinaryRef<'a> {
                    pub fn byte_width(&self) -> planus::Result<i32> {
                        Ok(self
                            .0
                            .access(0, "FixedSizeBinary", "byte_width")?
                            .unwrap_or(0))
                    }
                }

                impl<'a> std::fmt::Debug for FixedSizeBinaryRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("FixedSizeBinaryRef");
                        if let Ok(byte_width) = self.byte_width() {
                            f.field("byte_width", &byte_width);
                        }
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for FixedSizeBinaryRef<'a> {
                    type Value = FixedSizeBinary;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(FixedSizeBinary {
                            byte_width: planus::ToOwned::to_owned(self.byte_width()?)?,
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for FixedSizeBinaryRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for FixedSizeBinary {
                    type Output = planus::Result<FixedSizeBinaryRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[FixedSizeBinaryRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<FixedSizeBinary>> for FixedSizeBinary {
                    type Value = planus::Offset<FixedSizeBinary>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<FixedSizeBinary>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for FixedSizeBinaryRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[FixedSizeBinaryRef]",
                                "read_as_root",
                                0,
                            )
                        })
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct Bool {}

                impl Bool {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(builder: &mut planus::Builder) -> planus::Offset<Self> {
                        let mut table_writer =
                            planus::table_writer::TableWriter::<4, 0>::new(builder);

                        table_writer.finish_calculating();

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<Bool>> for Bool {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Bool> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<Bool>> for Bool {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<Bool>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<Bool> for Bool {
                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Bool> {
                        Bool::create(builder)
                    }
                }

                #[derive(Copy, Clone)]
                pub struct BoolRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> BoolRef<'a> {}

                impl<'a> std::fmt::Debug for BoolRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("BoolRef");

                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for BoolRef<'a> {
                    type Value = Bool;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(Bool {})
                    }
                }

                impl<'a> planus::TableRead<'a> for BoolRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for Bool {
                    type Output = planus::Result<BoolRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[BoolRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<Bool>> for Bool {
                    type Value = planus::Offset<Bool>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<Bool>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for BoolRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location("[BoolRef]", "read_as_root", 0)
                        })
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct Decimal {
                    pub precision: i32,
                    pub scale: i32,
                    pub bit_width: i32,
                }

                impl Decimal {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(
                        builder: &mut planus::Builder,
                        precision: impl planus::WriteAsDefault<i32, i32>,
                        scale: impl planus::WriteAsDefault<i32, i32>,
                        bit_width: impl planus::WriteAsDefault<i32, i32>,
                    ) -> planus::Offset<Self> {
                        let prepared_precision = precision.prepare(builder, &0);

                        let prepared_scale = scale.prepare(builder, &0);

                        let prepared_bit_width = bit_width.prepare(builder, &128);

                        let mut table_writer =
                            planus::table_writer::TableWriter::<8, 12>::new(builder);

                        if prepared_precision.is_some() {
                            table_writer.calculate_size::<i32>(2);
                        }
                        if prepared_scale.is_some() {
                            table_writer.calculate_size::<i32>(4);
                        }
                        if prepared_bit_width.is_some() {
                            table_writer.calculate_size::<i32>(6);
                        }

                        table_writer.finish_calculating();

                        unsafe {
                            if let Some(prepared_precision) = prepared_precision {
                                table_writer.write::<_, _, 4>(0, &prepared_precision);
                            }
                            if let Some(prepared_scale) = prepared_scale {
                                table_writer.write::<_, _, 4>(1, &prepared_scale);
                            }
                            if let Some(prepared_bit_width) = prepared_bit_width {
                                table_writer.write::<_, _, 4>(2, &prepared_bit_width);
                            }
                        }

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<Decimal>> for Decimal {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Decimal> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<Decimal>> for Decimal {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<Decimal>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<Decimal> for Decimal {
                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Decimal> {
                        Decimal::create(builder, &self.precision, &self.scale, &self.bit_width)
                    }
                }

                #[derive(Copy, Clone)]
                pub struct DecimalRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> DecimalRef<'a> {
                    pub fn precision(&self) -> planus::Result<i32> {
                        Ok(self.0.access(0, "Decimal", "precision")?.unwrap_or(0))
                    }

                    pub fn scale(&self) -> planus::Result<i32> {
                        Ok(self.0.access(1, "Decimal", "scale")?.unwrap_or(0))
                    }

                    pub fn bit_width(&self) -> planus::Result<i32> {
                        Ok(self.0.access(2, "Decimal", "bit_width")?.unwrap_or(128))
                    }
                }

                impl<'a> std::fmt::Debug for DecimalRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("DecimalRef");
                        if let Ok(precision) = self.precision() {
                            f.field("precision", &precision);
                        }
                        if let Ok(scale) = self.scale() {
                            f.field("scale", &scale);
                        }
                        if let Ok(bit_width) = self.bit_width() {
                            f.field("bit_width", &bit_width);
                        }
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for DecimalRef<'a> {
                    type Value = Decimal;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(Decimal {
                            precision: planus::ToOwned::to_owned(self.precision()?)?,
                            scale: planus::ToOwned::to_owned(self.scale()?)?,
                            bit_width: planus::ToOwned::to_owned(self.bit_width()?)?,
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for DecimalRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for Decimal {
                    type Output = planus::Result<DecimalRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[DecimalRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<Decimal>> for Decimal {
                    type Value = planus::Offset<Decimal>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<Decimal>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for DecimalRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location("[DecimalRef]", "read_as_root", 0)
                        })
                    }
                }

                #[derive(
                    Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize,
                )]
                #[repr(i16)]
                pub enum DateUnit {
                    Day = 0,
                    Millisecond = 1,
                }

                impl TryFrom<i16> for DateUnit {
                    type Error = planus::errors::UnknownEnumTagKind;
                    fn try_from(
                        value: i16,
                    ) -> std::result::Result<Self, planus::errors::UnknownEnumTagKind>
                    {
                        #[allow(clippy::match_single_binding)]
                        match value {
                            0 => Ok(DateUnit::Day),
                            1 => Ok(DateUnit::Millisecond),

                            _ => Err(planus::errors::UnknownEnumTagKind { tag: value as i128 }),
                        }
                    }
                }

                impl From<DateUnit> for i16 {
                    fn from(value: DateUnit) -> Self {
                        value as i16
                    }
                }

                impl planus::ToOwned for DateUnit {
                    type Value = DateUnit;
                    #[inline]
                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(self)
                    }
                }

                impl planus::Primitive for DateUnit {
                    const ALIGNMENT: usize = 2;
                    const SIZE: usize = 2;
                }

                impl planus::WriteAsPrimitive<DateUnit> for DateUnit {
                    #[inline]
                    fn write<const N: usize>(
                        &self,
                        cursor: planus::Cursor<'_, N>,
                        buffer_position: u32,
                    ) {
                        (*self as i16).write(cursor, buffer_position);
                    }
                }

                impl planus::WriteAs<DateUnit> for DateUnit {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(&self, _builder: &mut planus::Builder) -> DateUnit {
                        *self
                    }
                }

                impl planus::WriteAsDefault<DateUnit, DateUnit> for DateUnit {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(
                        &self,
                        _builder: &mut planus::Builder,
                        default: &DateUnit,
                    ) -> Option<DateUnit> {
                        if self == default {
                            None
                        } else {
                            Some(*self)
                        }
                    }
                }

                impl planus::WriteAsOptional<DateUnit> for DateUnit {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(&self, _builder: &mut planus::Builder) -> Option<DateUnit> {
                        Some(*self)
                    }
                }

                impl<'buf> planus::TableRead<'buf> for DateUnit {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'buf>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        let n: i16 = planus::TableRead::from_buffer(buffer, offset)?;
                        Ok(n.try_into()?)
                    }
                }

                impl<'buf> planus::VectorRead<'buf> for DateUnit {
                    type Output = std::result::Result<Self, planus::errors::UnknownEnumTag>;

                    const STRIDE: usize = 2;
                    #[inline]
                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'buf>,
                        offset: usize,
                    ) -> Self::Output {
                        let value = <i16 as planus::VectorRead>::from_buffer(buffer, offset);
                        let value: std::result::Result<Self, _> = value.try_into();
                        value.map_err(|error_kind| {
                            error_kind.with_error_location(
                                "DateUnit",
                                "VectorRead::from_buffer",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl<'buf> planus::VectorWrite<DateUnit> for DateUnit {
                    const STRIDE: usize = 2;

                    type Value = Self;

                    fn prepare(&self, _builder: &mut planus::Builder) -> Self::Value {
                        *self
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[Self],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 2];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (2 * i) as u32,
                            );
                        }
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct Date {
                    pub unit: self::DateUnit,
                }

                impl Date {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(
                        builder: &mut planus::Builder,
                        unit: impl planus::WriteAsDefault<self::DateUnit, self::DateUnit>,
                    ) -> planus::Offset<Self> {
                        let prepared_unit = unit.prepare(builder, &self::DateUnit::Millisecond);

                        let mut table_writer =
                            planus::table_writer::TableWriter::<4, 2>::new(builder);

                        if prepared_unit.is_some() {
                            table_writer.calculate_size::<self::DateUnit>(2);
                        }

                        table_writer.finish_calculating();

                        unsafe {
                            if let Some(prepared_unit) = prepared_unit {
                                table_writer.write::<_, _, 2>(0, &prepared_unit);
                            }
                        }

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<Date>> for Date {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Date> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<Date>> for Date {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<Date>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<Date> for Date {
                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Date> {
                        Date::create(builder, &self.unit)
                    }
                }

                #[derive(Copy, Clone)]
                pub struct DateRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> DateRef<'a> {
                    pub fn unit(&self) -> planus::Result<self::DateUnit> {
                        Ok(self
                            .0
                            .access(0, "Date", "unit")?
                            .unwrap_or(self::DateUnit::Millisecond))
                    }
                }

                impl<'a> std::fmt::Debug for DateRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("DateRef");
                        if let Ok(unit) = self.unit() {
                            f.field("unit", &unit);
                        }
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for DateRef<'a> {
                    type Value = Date;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(Date {
                            unit: planus::ToOwned::to_owned(self.unit()?)?,
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for DateRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for Date {
                    type Output = planus::Result<DateRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[DateRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<Date>> for Date {
                    type Value = planus::Offset<Date>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<Date>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for DateRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location("[DateRef]", "read_as_root", 0)
                        })
                    }
                }

                #[derive(
                    Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize,
                )]
                #[repr(i16)]
                pub enum TimeUnit {
                    Second = 0,
                    Millisecond = 1,
                    Microsecond = 2,
                    Nanosecond = 3,
                }

                impl TryFrom<i16> for TimeUnit {
                    type Error = planus::errors::UnknownEnumTagKind;
                    fn try_from(
                        value: i16,
                    ) -> std::result::Result<Self, planus::errors::UnknownEnumTagKind>
                    {
                        #[allow(clippy::match_single_binding)]
                        match value {
                            0 => Ok(TimeUnit::Second),
                            1 => Ok(TimeUnit::Millisecond),
                            2 => Ok(TimeUnit::Microsecond),
                            3 => Ok(TimeUnit::Nanosecond),

                            _ => Err(planus::errors::UnknownEnumTagKind { tag: value as i128 }),
                        }
                    }
                }

                impl From<TimeUnit> for i16 {
                    fn from(value: TimeUnit) -> Self {
                        value as i16
                    }
                }

                impl planus::ToOwned for TimeUnit {
                    type Value = TimeUnit;
                    #[inline]
                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(self)
                    }
                }

                impl planus::Primitive for TimeUnit {
                    const ALIGNMENT: usize = 2;
                    const SIZE: usize = 2;
                }

                impl planus::WriteAsPrimitive<TimeUnit> for TimeUnit {
                    #[inline]
                    fn write<const N: usize>(
                        &self,
                        cursor: planus::Cursor<'_, N>,
                        buffer_position: u32,
                    ) {
                        (*self as i16).write(cursor, buffer_position);
                    }
                }

                impl planus::WriteAs<TimeUnit> for TimeUnit {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(&self, _builder: &mut planus::Builder) -> TimeUnit {
                        *self
                    }
                }

                impl planus::WriteAsDefault<TimeUnit, TimeUnit> for TimeUnit {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(
                        &self,
                        _builder: &mut planus::Builder,
                        default: &TimeUnit,
                    ) -> Option<TimeUnit> {
                        if self == default {
                            None
                        } else {
                            Some(*self)
                        }
                    }
                }

                impl planus::WriteAsOptional<TimeUnit> for TimeUnit {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(&self, _builder: &mut planus::Builder) -> Option<TimeUnit> {
                        Some(*self)
                    }
                }

                impl<'buf> planus::TableRead<'buf> for TimeUnit {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'buf>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        let n: i16 = planus::TableRead::from_buffer(buffer, offset)?;
                        Ok(n.try_into()?)
                    }
                }

                impl<'buf> planus::VectorRead<'buf> for TimeUnit {
                    type Output = std::result::Result<Self, planus::errors::UnknownEnumTag>;

                    const STRIDE: usize = 2;
                    #[inline]
                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'buf>,
                        offset: usize,
                    ) -> Self::Output {
                        let value = <i16 as planus::VectorRead>::from_buffer(buffer, offset);
                        let value: std::result::Result<Self, _> = value.try_into();
                        value.map_err(|error_kind| {
                            error_kind.with_error_location(
                                "TimeUnit",
                                "VectorRead::from_buffer",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl<'buf> planus::VectorWrite<TimeUnit> for TimeUnit {
                    const STRIDE: usize = 2;

                    type Value = Self;

                    fn prepare(&self, _builder: &mut planus::Builder) -> Self::Value {
                        *self
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[Self],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 2];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (2 * i) as u32,
                            );
                        }
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct Time {
                    pub unit: self::TimeUnit,
                    pub bit_width: i32,
                }

                impl Time {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(
                        builder: &mut planus::Builder,
                        unit: impl planus::WriteAsDefault<self::TimeUnit, self::TimeUnit>,
                        bit_width: impl planus::WriteAsDefault<i32, i32>,
                    ) -> planus::Offset<Self> {
                        let prepared_unit = unit.prepare(builder, &self::TimeUnit::Millisecond);

                        let prepared_bit_width = bit_width.prepare(builder, &32);

                        let mut table_writer =
                            planus::table_writer::TableWriter::<6, 6>::new(builder);

                        if prepared_unit.is_some() {
                            table_writer.calculate_size::<self::TimeUnit>(2);
                        }
                        if prepared_bit_width.is_some() {
                            table_writer.calculate_size::<i32>(4);
                        }

                        table_writer.finish_calculating();

                        unsafe {
                            if let Some(prepared_bit_width) = prepared_bit_width {
                                table_writer.write::<_, _, 4>(1, &prepared_bit_width);
                            }
                            if let Some(prepared_unit) = prepared_unit {
                                table_writer.write::<_, _, 2>(0, &prepared_unit);
                            }
                        }

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<Time>> for Time {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Time> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<Time>> for Time {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<Time>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<Time> for Time {
                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Time> {
                        Time::create(builder, &self.unit, &self.bit_width)
                    }
                }

                #[derive(Copy, Clone)]
                pub struct TimeRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> TimeRef<'a> {
                    pub fn unit(&self) -> planus::Result<self::TimeUnit> {
                        Ok(self
                            .0
                            .access(0, "Time", "unit")?
                            .unwrap_or(self::TimeUnit::Millisecond))
                    }

                    pub fn bit_width(&self) -> planus::Result<i32> {
                        Ok(self.0.access(1, "Time", "bit_width")?.unwrap_or(32))
                    }
                }

                impl<'a> std::fmt::Debug for TimeRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("TimeRef");
                        if let Ok(unit) = self.unit() {
                            f.field("unit", &unit);
                        }
                        if let Ok(bit_width) = self.bit_width() {
                            f.field("bit_width", &bit_width);
                        }
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for TimeRef<'a> {
                    type Value = Time;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(Time {
                            unit: planus::ToOwned::to_owned(self.unit()?)?,
                            bit_width: planus::ToOwned::to_owned(self.bit_width()?)?,
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for TimeRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for Time {
                    type Output = planus::Result<TimeRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[TimeRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<Time>> for Time {
                    type Value = planus::Offset<Time>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<Time>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for TimeRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location("[TimeRef]", "read_as_root", 0)
                        })
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct Timestamp {
                    pub unit: self::TimeUnit,
                    pub timezone: Option<String>,
                }

                impl Timestamp {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(
                        builder: &mut planus::Builder,
                        unit: impl planus::WriteAsDefault<self::TimeUnit, self::TimeUnit>,
                        timezone: impl planus::WriteAsOptional<planus::Offset<str>>,
                    ) -> planus::Offset<Self> {
                        let prepared_unit = unit.prepare(builder, &self::TimeUnit::Second);

                        let prepared_timezone = timezone.prepare(builder);

                        let mut table_writer =
                            planus::table_writer::TableWriter::<6, 6>::new(builder);

                        if prepared_unit.is_some() {
                            table_writer.calculate_size::<self::TimeUnit>(2);
                        }
                        if prepared_timezone.is_some() {
                            table_writer.calculate_size::<planus::Offset<str>>(4);
                        }

                        table_writer.finish_calculating();

                        unsafe {
                            if let Some(prepared_timezone) = prepared_timezone {
                                table_writer.write::<_, _, 4>(1, &prepared_timezone);
                            }
                            if let Some(prepared_unit) = prepared_unit {
                                table_writer.write::<_, _, 2>(0, &prepared_unit);
                            }
                        }

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<Timestamp>> for Timestamp {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Timestamp> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<Timestamp>> for Timestamp {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<Timestamp>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<Timestamp> for Timestamp {
                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Timestamp> {
                        Timestamp::create(builder, &self.unit, &self.timezone)
                    }
                }

                #[derive(Copy, Clone)]
                pub struct TimestampRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> TimestampRef<'a> {
                    pub fn unit(&self) -> planus::Result<self::TimeUnit> {
                        Ok(self
                            .0
                            .access(0, "Timestamp", "unit")?
                            .unwrap_or(self::TimeUnit::Second))
                    }

                    pub fn timezone(&self) -> planus::Result<Option<&'a str>> {
                        self.0.access(1, "Timestamp", "timezone")
                    }
                }

                impl<'a> std::fmt::Debug for TimestampRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("TimestampRef");
                        if let Ok(unit) = self.unit() {
                            f.field("unit", &unit);
                        }
                        if let Ok(Some(timezone)) = self.timezone() {
                            f.field("timezone", &timezone);
                        }
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for TimestampRef<'a> {
                    type Value = Timestamp;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(Timestamp {
                            unit: planus::ToOwned::to_owned(self.unit()?)?,
                            timezone: if let Some(timezone) = self.timezone()? {
                                Some(planus::ToOwned::to_owned(timezone)?)
                            } else {
                                None
                            },
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for TimestampRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for Timestamp {
                    type Output = planus::Result<TimestampRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[TimestampRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<Timestamp>> for Timestamp {
                    type Value = planus::Offset<Timestamp>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<Timestamp>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for TimestampRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location("[TimestampRef]", "read_as_root", 0)
                        })
                    }
                }

                #[derive(
                    Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize,
                )]
                #[repr(i16)]
                pub enum IntervalUnit {
                    YearMonth = 0,
                    DayTime = 1,
                    MonthDayNano = 2,
                }

                impl TryFrom<i16> for IntervalUnit {
                    type Error = planus::errors::UnknownEnumTagKind;
                    fn try_from(
                        value: i16,
                    ) -> std::result::Result<Self, planus::errors::UnknownEnumTagKind>
                    {
                        #[allow(clippy::match_single_binding)]
                        match value {
                            0 => Ok(IntervalUnit::YearMonth),
                            1 => Ok(IntervalUnit::DayTime),
                            2 => Ok(IntervalUnit::MonthDayNano),

                            _ => Err(planus::errors::UnknownEnumTagKind { tag: value as i128 }),
                        }
                    }
                }

                impl From<IntervalUnit> for i16 {
                    fn from(value: IntervalUnit) -> Self {
                        value as i16
                    }
                }

                impl planus::ToOwned for IntervalUnit {
                    type Value = IntervalUnit;
                    #[inline]
                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(self)
                    }
                }

                impl planus::Primitive for IntervalUnit {
                    const ALIGNMENT: usize = 2;
                    const SIZE: usize = 2;
                }

                impl planus::WriteAsPrimitive<IntervalUnit> for IntervalUnit {
                    #[inline]
                    fn write<const N: usize>(
                        &self,
                        cursor: planus::Cursor<'_, N>,
                        buffer_position: u32,
                    ) {
                        (*self as i16).write(cursor, buffer_position);
                    }
                }

                impl planus::WriteAs<IntervalUnit> for IntervalUnit {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(&self, _builder: &mut planus::Builder) -> IntervalUnit {
                        *self
                    }
                }

                impl planus::WriteAsDefault<IntervalUnit, IntervalUnit> for IntervalUnit {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(
                        &self,
                        _builder: &mut planus::Builder,
                        default: &IntervalUnit,
                    ) -> Option<IntervalUnit> {
                        if self == default {
                            None
                        } else {
                            Some(*self)
                        }
                    }
                }

                impl planus::WriteAsOptional<IntervalUnit> for IntervalUnit {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(&self, _builder: &mut planus::Builder) -> Option<IntervalUnit> {
                        Some(*self)
                    }
                }

                impl<'buf> planus::TableRead<'buf> for IntervalUnit {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'buf>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        let n: i16 = planus::TableRead::from_buffer(buffer, offset)?;
                        Ok(n.try_into()?)
                    }
                }

                impl<'buf> planus::VectorRead<'buf> for IntervalUnit {
                    type Output = std::result::Result<Self, planus::errors::UnknownEnumTag>;

                    const STRIDE: usize = 2;
                    #[inline]
                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'buf>,
                        offset: usize,
                    ) -> Self::Output {
                        let value = <i16 as planus::VectorRead>::from_buffer(buffer, offset);
                        let value: std::result::Result<Self, _> = value.try_into();
                        value.map_err(|error_kind| {
                            error_kind.with_error_location(
                                "IntervalUnit",
                                "VectorRead::from_buffer",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl<'buf> planus::VectorWrite<IntervalUnit> for IntervalUnit {
                    const STRIDE: usize = 2;

                    type Value = Self;

                    fn prepare(&self, _builder: &mut planus::Builder) -> Self::Value {
                        *self
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[Self],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 2];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (2 * i) as u32,
                            );
                        }
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct Interval {
                    pub unit: self::IntervalUnit,
                }

                impl Interval {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(
                        builder: &mut planus::Builder,
                        unit: impl planus::WriteAsDefault<self::IntervalUnit, self::IntervalUnit>,
                    ) -> planus::Offset<Self> {
                        let prepared_unit = unit.prepare(builder, &self::IntervalUnit::YearMonth);

                        let mut table_writer =
                            planus::table_writer::TableWriter::<4, 2>::new(builder);

                        if prepared_unit.is_some() {
                            table_writer.calculate_size::<self::IntervalUnit>(2);
                        }

                        table_writer.finish_calculating();

                        unsafe {
                            if let Some(prepared_unit) = prepared_unit {
                                table_writer.write::<_, _, 2>(0, &prepared_unit);
                            }
                        }

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<Interval>> for Interval {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Interval> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<Interval>> for Interval {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<Interval>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<Interval> for Interval {
                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Interval> {
                        Interval::create(builder, &self.unit)
                    }
                }

                #[derive(Copy, Clone)]
                pub struct IntervalRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> IntervalRef<'a> {
                    pub fn unit(&self) -> planus::Result<self::IntervalUnit> {
                        Ok(self
                            .0
                            .access(0, "Interval", "unit")?
                            .unwrap_or(self::IntervalUnit::YearMonth))
                    }
                }

                impl<'a> std::fmt::Debug for IntervalRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("IntervalRef");
                        if let Ok(unit) = self.unit() {
                            f.field("unit", &unit);
                        }
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for IntervalRef<'a> {
                    type Value = Interval;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(Interval {
                            unit: planus::ToOwned::to_owned(self.unit()?)?,
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for IntervalRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for Interval {
                    type Output = planus::Result<IntervalRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[IntervalRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<Interval>> for Interval {
                    type Value = planus::Offset<Interval>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<Interval>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for IntervalRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location("[IntervalRef]", "read_as_root", 0)
                        })
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct Duration {
                    pub unit: self::TimeUnit,
                }

                impl Duration {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(
                        builder: &mut planus::Builder,
                        unit: impl planus::WriteAsDefault<self::TimeUnit, self::TimeUnit>,
                    ) -> planus::Offset<Self> {
                        let prepared_unit = unit.prepare(builder, &self::TimeUnit::Millisecond);

                        let mut table_writer =
                            planus::table_writer::TableWriter::<4, 2>::new(builder);

                        if prepared_unit.is_some() {
                            table_writer.calculate_size::<self::TimeUnit>(2);
                        }

                        table_writer.finish_calculating();

                        unsafe {
                            if let Some(prepared_unit) = prepared_unit {
                                table_writer.write::<_, _, 2>(0, &prepared_unit);
                            }
                        }

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<Duration>> for Duration {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Duration> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<Duration>> for Duration {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<Duration>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<Duration> for Duration {
                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Duration> {
                        Duration::create(builder, &self.unit)
                    }
                }

                #[derive(Copy, Clone)]
                pub struct DurationRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> DurationRef<'a> {
                    pub fn unit(&self) -> planus::Result<self::TimeUnit> {
                        Ok(self
                            .0
                            .access(0, "Duration", "unit")?
                            .unwrap_or(self::TimeUnit::Millisecond))
                    }
                }

                impl<'a> std::fmt::Debug for DurationRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("DurationRef");
                        if let Ok(unit) = self.unit() {
                            f.field("unit", &unit);
                        }
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for DurationRef<'a> {
                    type Value = Duration;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(Duration {
                            unit: planus::ToOwned::to_owned(self.unit()?)?,
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for DurationRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for Duration {
                    type Output = planus::Result<DurationRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[DurationRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<Duration>> for Duration {
                    type Value = planus::Offset<Duration>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<Duration>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for DurationRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location("[DurationRef]", "read_as_root", 0)
                        })
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub enum Type {
                    Null(Box<self::Null>),
                    Int(Box<self::Int>),
                    FloatingPoint(Box<self::FloatingPoint>),
                    Binary(Box<self::Binary>),
                    Utf8(Box<self::Utf8>),
                    Bool(Box<self::Bool>),
                    Decimal(Box<self::Decimal>),
                    Date(Box<self::Date>),
                    Time(Box<self::Time>),
                    Timestamp(Box<self::Timestamp>),
                    Interval(Box<self::Interval>),
                    List(Box<self::List>),
                    Struct(Box<self::Struct>),
                    Union(Box<self::Union>),
                    FixedSizeBinary(Box<self::FixedSizeBinary>),
                    FixedSizeList(Box<self::FixedSizeList>),
                    Map(Box<self::Map>),
                    Duration(Box<self::Duration>),
                    LargeBinary(Box<self::LargeBinary>),
                    LargeUtf8(Box<self::LargeUtf8>),
                    LargeList(Box<self::LargeList>),
                }

                impl Type {
                    pub fn create_null(
                        builder: &mut planus::Builder,
                        value: impl planus::WriteAsOffset<self::Null>,
                    ) -> planus::UnionOffset<Self> {
                        planus::UnionOffset::new(1, value.prepare(builder).downcast())
                    }

                    pub fn create_int(
                        builder: &mut planus::Builder,
                        value: impl planus::WriteAsOffset<self::Int>,
                    ) -> planus::UnionOffset<Self> {
                        planus::UnionOffset::new(2, value.prepare(builder).downcast())
                    }

                    pub fn create_floating_point(
                        builder: &mut planus::Builder,
                        value: impl planus::WriteAsOffset<self::FloatingPoint>,
                    ) -> planus::UnionOffset<Self> {
                        planus::UnionOffset::new(3, value.prepare(builder).downcast())
                    }

                    pub fn create_binary(
                        builder: &mut planus::Builder,
                        value: impl planus::WriteAsOffset<self::Binary>,
                    ) -> planus::UnionOffset<Self> {
                        planus::UnionOffset::new(4, value.prepare(builder).downcast())
                    }

                    pub fn create_utf8(
                        builder: &mut planus::Builder,
                        value: impl planus::WriteAsOffset<self::Utf8>,
                    ) -> planus::UnionOffset<Self> {
                        planus::UnionOffset::new(5, value.prepare(builder).downcast())
                    }

                    pub fn create_bool(
                        builder: &mut planus::Builder,
                        value: impl planus::WriteAsOffset<self::Bool>,
                    ) -> planus::UnionOffset<Self> {
                        planus::UnionOffset::new(6, value.prepare(builder).downcast())
                    }

                    pub fn create_decimal(
                        builder: &mut planus::Builder,
                        value: impl planus::WriteAsOffset<self::Decimal>,
                    ) -> planus::UnionOffset<Self> {
                        planus::UnionOffset::new(7, value.prepare(builder).downcast())
                    }

                    pub fn create_date(
                        builder: &mut planus::Builder,
                        value: impl planus::WriteAsOffset<self::Date>,
                    ) -> planus::UnionOffset<Self> {
                        planus::UnionOffset::new(8, value.prepare(builder).downcast())
                    }

                    pub fn create_time(
                        builder: &mut planus::Builder,
                        value: impl planus::WriteAsOffset<self::Time>,
                    ) -> planus::UnionOffset<Self> {
                        planus::UnionOffset::new(9, value.prepare(builder).downcast())
                    }

                    pub fn create_timestamp(
                        builder: &mut planus::Builder,
                        value: impl planus::WriteAsOffset<self::Timestamp>,
                    ) -> planus::UnionOffset<Self> {
                        planus::UnionOffset::new(10, value.prepare(builder).downcast())
                    }

                    pub fn create_interval(
                        builder: &mut planus::Builder,
                        value: impl planus::WriteAsOffset<self::Interval>,
                    ) -> planus::UnionOffset<Self> {
                        planus::UnionOffset::new(11, value.prepare(builder).downcast())
                    }

                    pub fn create_list(
                        builder: &mut planus::Builder,
                        value: impl planus::WriteAsOffset<self::List>,
                    ) -> planus::UnionOffset<Self> {
                        planus::UnionOffset::new(12, value.prepare(builder).downcast())
                    }

                    pub fn create_struct(
                        builder: &mut planus::Builder,
                        value: impl planus::WriteAsOffset<self::Struct>,
                    ) -> planus::UnionOffset<Self> {
                        planus::UnionOffset::new(13, value.prepare(builder).downcast())
                    }

                    pub fn create_union(
                        builder: &mut planus::Builder,
                        value: impl planus::WriteAsOffset<self::Union>,
                    ) -> planus::UnionOffset<Self> {
                        planus::UnionOffset::new(14, value.prepare(builder).downcast())
                    }

                    pub fn create_fixed_size_binary(
                        builder: &mut planus::Builder,
                        value: impl planus::WriteAsOffset<self::FixedSizeBinary>,
                    ) -> planus::UnionOffset<Self> {
                        planus::UnionOffset::new(15, value.prepare(builder).downcast())
                    }

                    pub fn create_fixed_size_list(
                        builder: &mut planus::Builder,
                        value: impl planus::WriteAsOffset<self::FixedSizeList>,
                    ) -> planus::UnionOffset<Self> {
                        planus::UnionOffset::new(16, value.prepare(builder).downcast())
                    }

                    pub fn create_map(
                        builder: &mut planus::Builder,
                        value: impl planus::WriteAsOffset<self::Map>,
                    ) -> planus::UnionOffset<Self> {
                        planus::UnionOffset::new(17, value.prepare(builder).downcast())
                    }

                    pub fn create_duration(
                        builder: &mut planus::Builder,
                        value: impl planus::WriteAsOffset<self::Duration>,
                    ) -> planus::UnionOffset<Self> {
                        planus::UnionOffset::new(18, value.prepare(builder).downcast())
                    }

                    pub fn create_large_binary(
                        builder: &mut planus::Builder,
                        value: impl planus::WriteAsOffset<self::LargeBinary>,
                    ) -> planus::UnionOffset<Self> {
                        planus::UnionOffset::new(19, value.prepare(builder).downcast())
                    }

                    pub fn create_large_utf8(
                        builder: &mut planus::Builder,
                        value: impl planus::WriteAsOffset<self::LargeUtf8>,
                    ) -> planus::UnionOffset<Self> {
                        planus::UnionOffset::new(20, value.prepare(builder).downcast())
                    }

                    pub fn create_large_list(
                        builder: &mut planus::Builder,
                        value: impl planus::WriteAsOffset<self::LargeList>,
                    ) -> planus::UnionOffset<Self> {
                        planus::UnionOffset::new(21, value.prepare(builder).downcast())
                    }
                }

                impl planus::WriteAsUnion<Type> for Type {
                    fn prepare(&self, builder: &mut planus::Builder) -> planus::UnionOffset<Self> {
                        match self {
                            Self::Null(value) => Self::create_null(builder, value),
                            Self::Int(value) => Self::create_int(builder, value),
                            Self::FloatingPoint(value) => {
                                Self::create_floating_point(builder, value)
                            }
                            Self::Binary(value) => Self::create_binary(builder, value),
                            Self::Utf8(value) => Self::create_utf8(builder, value),
                            Self::Bool(value) => Self::create_bool(builder, value),
                            Self::Decimal(value) => Self::create_decimal(builder, value),
                            Self::Date(value) => Self::create_date(builder, value),
                            Self::Time(value) => Self::create_time(builder, value),
                            Self::Timestamp(value) => Self::create_timestamp(builder, value),
                            Self::Interval(value) => Self::create_interval(builder, value),
                            Self::List(value) => Self::create_list(builder, value),
                            Self::Struct(value) => Self::create_struct(builder, value),
                            Self::Union(value) => Self::create_union(builder, value),
                            Self::FixedSizeBinary(value) => {
                                Self::create_fixed_size_binary(builder, value)
                            }
                            Self::FixedSizeList(value) => {
                                Self::create_fixed_size_list(builder, value)
                            }
                            Self::Map(value) => Self::create_map(builder, value),
                            Self::Duration(value) => Self::create_duration(builder, value),
                            Self::LargeBinary(value) => Self::create_large_binary(builder, value),
                            Self::LargeUtf8(value) => Self::create_large_utf8(builder, value),
                            Self::LargeList(value) => Self::create_large_list(builder, value),
                        }
                    }
                }

                impl planus::WriteAsOptionalUnion<Type> for Type {
                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::UnionOffset<Self>> {
                        Some(planus::WriteAsUnion::prepare(self, builder))
                    }
                }

                #[derive(Copy, Clone, Debug)]
                pub enum TypeRef<'a> {
                    Null(self::NullRef<'a>),
                    Int(self::IntRef<'a>),
                    FloatingPoint(self::FloatingPointRef<'a>),
                    Binary(self::BinaryRef<'a>),
                    Utf8(self::Utf8Ref<'a>),
                    Bool(self::BoolRef<'a>),
                    Decimal(self::DecimalRef<'a>),
                    Date(self::DateRef<'a>),
                    Time(self::TimeRef<'a>),
                    Timestamp(self::TimestampRef<'a>),
                    Interval(self::IntervalRef<'a>),
                    List(self::ListRef<'a>),
                    Struct(self::StructRef<'a>),
                    Union(self::UnionRef<'a>),
                    FixedSizeBinary(self::FixedSizeBinaryRef<'a>),
                    FixedSizeList(self::FixedSizeListRef<'a>),
                    Map(self::MapRef<'a>),
                    Duration(self::DurationRef<'a>),
                    LargeBinary(self::LargeBinaryRef<'a>),
                    LargeUtf8(self::LargeUtf8Ref<'a>),
                    LargeList(self::LargeListRef<'a>),
                }

                impl<'a> planus::ToOwned for TypeRef<'a> {
                    type Value = Type;

                    fn to_owned(self) -> planus::Result<Type> {
                        Ok(match self {
                            Self::Null(value) => {
                                Type::Null(Box::new(planus::ToOwned::to_owned(value)?))
                            }

                            Self::Int(value) => {
                                Type::Int(Box::new(planus::ToOwned::to_owned(value)?))
                            }

                            Self::FloatingPoint(value) => {
                                Type::FloatingPoint(Box::new(planus::ToOwned::to_owned(value)?))
                            }

                            Self::Binary(value) => {
                                Type::Binary(Box::new(planus::ToOwned::to_owned(value)?))
                            }

                            Self::Utf8(value) => {
                                Type::Utf8(Box::new(planus::ToOwned::to_owned(value)?))
                            }

                            Self::Bool(value) => {
                                Type::Bool(Box::new(planus::ToOwned::to_owned(value)?))
                            }

                            Self::Decimal(value) => {
                                Type::Decimal(Box::new(planus::ToOwned::to_owned(value)?))
                            }

                            Self::Date(value) => {
                                Type::Date(Box::new(planus::ToOwned::to_owned(value)?))
                            }

                            Self::Time(value) => {
                                Type::Time(Box::new(planus::ToOwned::to_owned(value)?))
                            }

                            Self::Timestamp(value) => {
                                Type::Timestamp(Box::new(planus::ToOwned::to_owned(value)?))
                            }

                            Self::Interval(value) => {
                                Type::Interval(Box::new(planus::ToOwned::to_owned(value)?))
                            }

                            Self::List(value) => {
                                Type::List(Box::new(planus::ToOwned::to_owned(value)?))
                            }

                            Self::Struct(value) => {
                                Type::Struct(Box::new(planus::ToOwned::to_owned(value)?))
                            }

                            Self::Union(value) => {
                                Type::Union(Box::new(planus::ToOwned::to_owned(value)?))
                            }

                            Self::FixedSizeBinary(value) => {
                                Type::FixedSizeBinary(Box::new(planus::ToOwned::to_owned(value)?))
                            }

                            Self::FixedSizeList(value) => {
                                Type::FixedSizeList(Box::new(planus::ToOwned::to_owned(value)?))
                            }

                            Self::Map(value) => {
                                Type::Map(Box::new(planus::ToOwned::to_owned(value)?))
                            }

                            Self::Duration(value) => {
                                Type::Duration(Box::new(planus::ToOwned::to_owned(value)?))
                            }

                            Self::LargeBinary(value) => {
                                Type::LargeBinary(Box::new(planus::ToOwned::to_owned(value)?))
                            }

                            Self::LargeUtf8(value) => {
                                Type::LargeUtf8(Box::new(planus::ToOwned::to_owned(value)?))
                            }

                            Self::LargeList(value) => {
                                Type::LargeList(Box::new(planus::ToOwned::to_owned(value)?))
                            }
                        })
                    }
                }

                impl<'a> planus::TableReadUnion<'a> for TypeRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        field_offset: usize,
                        tag: u8,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        match tag {
                            1 => Ok(Self::Null(planus::TableRead::from_buffer(
                                buffer,
                                field_offset,
                            )?)),
                            2 => Ok(Self::Int(planus::TableRead::from_buffer(
                                buffer,
                                field_offset,
                            )?)),
                            3 => Ok(Self::FloatingPoint(planus::TableRead::from_buffer(
                                buffer,
                                field_offset,
                            )?)),
                            4 => Ok(Self::Binary(planus::TableRead::from_buffer(
                                buffer,
                                field_offset,
                            )?)),
                            5 => Ok(Self::Utf8(planus::TableRead::from_buffer(
                                buffer,
                                field_offset,
                            )?)),
                            6 => Ok(Self::Bool(planus::TableRead::from_buffer(
                                buffer,
                                field_offset,
                            )?)),
                            7 => Ok(Self::Decimal(planus::TableRead::from_buffer(
                                buffer,
                                field_offset,
                            )?)),
                            8 => Ok(Self::Date(planus::TableRead::from_buffer(
                                buffer,
                                field_offset,
                            )?)),
                            9 => Ok(Self::Time(planus::TableRead::from_buffer(
                                buffer,
                                field_offset,
                            )?)),
                            10 => Ok(Self::Timestamp(planus::TableRead::from_buffer(
                                buffer,
                                field_offset,
                            )?)),
                            11 => Ok(Self::Interval(planus::TableRead::from_buffer(
                                buffer,
                                field_offset,
                            )?)),
                            12 => Ok(Self::List(planus::TableRead::from_buffer(
                                buffer,
                                field_offset,
                            )?)),
                            13 => Ok(Self::Struct(planus::TableRead::from_buffer(
                                buffer,
                                field_offset,
                            )?)),
                            14 => Ok(Self::Union(planus::TableRead::from_buffer(
                                buffer,
                                field_offset,
                            )?)),
                            15 => Ok(Self::FixedSizeBinary(planus::TableRead::from_buffer(
                                buffer,
                                field_offset,
                            )?)),
                            16 => Ok(Self::FixedSizeList(planus::TableRead::from_buffer(
                                buffer,
                                field_offset,
                            )?)),
                            17 => Ok(Self::Map(planus::TableRead::from_buffer(
                                buffer,
                                field_offset,
                            )?)),
                            18 => Ok(Self::Duration(planus::TableRead::from_buffer(
                                buffer,
                                field_offset,
                            )?)),
                            19 => Ok(Self::LargeBinary(planus::TableRead::from_buffer(
                                buffer,
                                field_offset,
                            )?)),
                            20 => Ok(Self::LargeUtf8(planus::TableRead::from_buffer(
                                buffer,
                                field_offset,
                            )?)),
                            21 => Ok(Self::LargeList(planus::TableRead::from_buffer(
                                buffer,
                                field_offset,
                            )?)),
                            _ => Err(planus::errors::ErrorKind::UnknownUnionTag { tag }),
                        }
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct KeyValue {
                    pub key: Option<String>,
                    pub value: Option<String>,
                }

                impl KeyValue {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(
                        builder: &mut planus::Builder,
                        key: impl planus::WriteAsOptional<planus::Offset<str>>,
                        value: impl planus::WriteAsOptional<planus::Offset<str>>,
                    ) -> planus::Offset<Self> {
                        let prepared_key = key.prepare(builder);

                        let prepared_value = value.prepare(builder);

                        let mut table_writer =
                            planus::table_writer::TableWriter::<6, 8>::new(builder);

                        if prepared_key.is_some() {
                            table_writer.calculate_size::<planus::Offset<str>>(2);
                        }
                        if prepared_value.is_some() {
                            table_writer.calculate_size::<planus::Offset<str>>(4);
                        }

                        table_writer.finish_calculating();

                        unsafe {
                            if let Some(prepared_key) = prepared_key {
                                table_writer.write::<_, _, 4>(0, &prepared_key);
                            }
                            if let Some(prepared_value) = prepared_value {
                                table_writer.write::<_, _, 4>(1, &prepared_value);
                            }
                        }

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<KeyValue>> for KeyValue {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<KeyValue> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<KeyValue>> for KeyValue {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<KeyValue>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<KeyValue> for KeyValue {
                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<KeyValue> {
                        KeyValue::create(builder, &self.key, &self.value)
                    }
                }

                #[derive(Copy, Clone)]
                pub struct KeyValueRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> KeyValueRef<'a> {
                    pub fn key(&self) -> planus::Result<Option<&'a str>> {
                        self.0.access(0, "KeyValue", "key")
                    }

                    pub fn value(&self) -> planus::Result<Option<&'a str>> {
                        self.0.access(1, "KeyValue", "value")
                    }
                }

                impl<'a> std::fmt::Debug for KeyValueRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("KeyValueRef");
                        if let Ok(Some(key)) = self.key() {
                            f.field("key", &key);
                        }
                        if let Ok(Some(value)) = self.value() {
                            f.field("value", &value);
                        }
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for KeyValueRef<'a> {
                    type Value = KeyValue;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(KeyValue {
                            key: if let Some(key) = self.key()? {
                                Some(planus::ToOwned::to_owned(key)?)
                            } else {
                                None
                            },
                            value: if let Some(value) = self.value()? {
                                Some(planus::ToOwned::to_owned(value)?)
                            } else {
                                None
                            },
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for KeyValueRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for KeyValue {
                    type Output = planus::Result<KeyValueRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[KeyValueRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<KeyValue>> for KeyValue {
                    type Value = planus::Offset<KeyValue>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<KeyValue>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for KeyValueRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location("[KeyValueRef]", "read_as_root", 0)
                        })
                    }
                }

                #[derive(
                    Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize,
                )]
                #[repr(i16)]
                pub enum DictionaryKind {
                    DenseArray = 0,
                }

                impl TryFrom<i16> for DictionaryKind {
                    type Error = planus::errors::UnknownEnumTagKind;
                    fn try_from(
                        value: i16,
                    ) -> std::result::Result<Self, planus::errors::UnknownEnumTagKind>
                    {
                        #[allow(clippy::match_single_binding)]
                        match value {
                            0 => Ok(DictionaryKind::DenseArray),

                            _ => Err(planus::errors::UnknownEnumTagKind { tag: value as i128 }),
                        }
                    }
                }

                impl From<DictionaryKind> for i16 {
                    fn from(value: DictionaryKind) -> Self {
                        value as i16
                    }
                }

                impl planus::ToOwned for DictionaryKind {
                    type Value = DictionaryKind;
                    #[inline]
                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(self)
                    }
                }

                impl planus::Primitive for DictionaryKind {
                    const ALIGNMENT: usize = 2;
                    const SIZE: usize = 2;
                }

                impl planus::WriteAsPrimitive<DictionaryKind> for DictionaryKind {
                    #[inline]
                    fn write<const N: usize>(
                        &self,
                        cursor: planus::Cursor<'_, N>,
                        buffer_position: u32,
                    ) {
                        (*self as i16).write(cursor, buffer_position);
                    }
                }

                impl planus::WriteAs<DictionaryKind> for DictionaryKind {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(&self, _builder: &mut planus::Builder) -> DictionaryKind {
                        *self
                    }
                }

                impl planus::WriteAsDefault<DictionaryKind, DictionaryKind> for DictionaryKind {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(
                        &self,
                        _builder: &mut planus::Builder,
                        default: &DictionaryKind,
                    ) -> Option<DictionaryKind> {
                        if self == default {
                            None
                        } else {
                            Some(*self)
                        }
                    }
                }

                impl planus::WriteAsOptional<DictionaryKind> for DictionaryKind {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(&self, _builder: &mut planus::Builder) -> Option<DictionaryKind> {
                        Some(*self)
                    }
                }

                impl<'buf> planus::TableRead<'buf> for DictionaryKind {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'buf>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        let n: i16 = planus::TableRead::from_buffer(buffer, offset)?;
                        Ok(n.try_into()?)
                    }
                }

                impl<'buf> planus::VectorRead<'buf> for DictionaryKind {
                    type Output = std::result::Result<Self, planus::errors::UnknownEnumTag>;

                    const STRIDE: usize = 2;
                    #[inline]
                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'buf>,
                        offset: usize,
                    ) -> Self::Output {
                        let value = <i16 as planus::VectorRead>::from_buffer(buffer, offset);
                        let value: std::result::Result<Self, _> = value.try_into();
                        value.map_err(|error_kind| {
                            error_kind.with_error_location(
                                "DictionaryKind",
                                "VectorRead::from_buffer",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl<'buf> planus::VectorWrite<DictionaryKind> for DictionaryKind {
                    const STRIDE: usize = 2;

                    type Value = Self;

                    fn prepare(&self, _builder: &mut planus::Builder) -> Self::Value {
                        *self
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[Self],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 2];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (2 * i) as u32,
                            );
                        }
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct DictionaryEncoding {
                    pub id: i64,
                    pub index_type: Option<Box<self::Int>>,
                    pub is_ordered: bool,
                    pub dictionary_kind: self::DictionaryKind,
                }

                impl DictionaryEncoding {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(
                        builder: &mut planus::Builder,
                        id: impl planus::WriteAsDefault<i64, i64>,
                        index_type: impl planus::WriteAsOptional<planus::Offset<self::Int>>,
                        is_ordered: impl planus::WriteAsDefault<bool, bool>,
                        dictionary_kind: impl planus::WriteAsDefault<
                            self::DictionaryKind,
                            self::DictionaryKind,
                        >,
                    ) -> planus::Offset<Self> {
                        let prepared_id = id.prepare(builder, &0);

                        let prepared_index_type = index_type.prepare(builder);

                        let prepared_is_ordered = is_ordered.prepare(builder, &false);

                        let prepared_dictionary_kind =
                            dictionary_kind.prepare(builder, &self::DictionaryKind::DenseArray);

                        let mut table_writer =
                            planus::table_writer::TableWriter::<10, 15>::new(builder);

                        if prepared_id.is_some() {
                            table_writer.calculate_size::<i64>(2);
                        }
                        if prepared_index_type.is_some() {
                            table_writer.calculate_size::<planus::Offset<self::Int>>(4);
                        }
                        if prepared_is_ordered.is_some() {
                            table_writer.calculate_size::<bool>(6);
                        }
                        if prepared_dictionary_kind.is_some() {
                            table_writer.calculate_size::<self::DictionaryKind>(8);
                        }

                        table_writer.finish_calculating();

                        unsafe {
                            if let Some(prepared_id) = prepared_id {
                                table_writer.write::<_, _, 8>(0, &prepared_id);
                            }
                            if let Some(prepared_index_type) = prepared_index_type {
                                table_writer.write::<_, _, 4>(1, &prepared_index_type);
                            }
                            if let Some(prepared_dictionary_kind) = prepared_dictionary_kind {
                                table_writer.write::<_, _, 2>(3, &prepared_dictionary_kind);
                            }
                            if let Some(prepared_is_ordered) = prepared_is_ordered {
                                table_writer.write::<_, _, 1>(2, &prepared_is_ordered);
                            }
                        }

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<DictionaryEncoding>> for DictionaryEncoding {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> planus::Offset<DictionaryEncoding> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<DictionaryEncoding>> for DictionaryEncoding {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<DictionaryEncoding>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<DictionaryEncoding> for DictionaryEncoding {
                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> planus::Offset<DictionaryEncoding> {
                        DictionaryEncoding::create(
                            builder,
                            &self.id,
                            &self.index_type,
                            &self.is_ordered,
                            &self.dictionary_kind,
                        )
                    }
                }

                #[derive(Copy, Clone)]
                pub struct DictionaryEncodingRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> DictionaryEncodingRef<'a> {
                    pub fn id(&self) -> planus::Result<i64> {
                        Ok(self.0.access(0, "DictionaryEncoding", "id")?.unwrap_or(0))
                    }

                    pub fn index_type(&self) -> planus::Result<Option<self::IntRef<'a>>> {
                        self.0.access(1, "DictionaryEncoding", "index_type")
                    }

                    pub fn is_ordered(&self) -> planus::Result<bool> {
                        Ok(self
                            .0
                            .access(2, "DictionaryEncoding", "is_ordered")?
                            .unwrap_or(false))
                    }

                    pub fn dictionary_kind(&self) -> planus::Result<self::DictionaryKind> {
                        Ok(self
                            .0
                            .access(3, "DictionaryEncoding", "dictionary_kind")?
                            .unwrap_or(self::DictionaryKind::DenseArray))
                    }
                }

                impl<'a> std::fmt::Debug for DictionaryEncodingRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("DictionaryEncodingRef");
                        if let Ok(id) = self.id() {
                            f.field("id", &id);
                        }
                        if let Ok(Some(index_type)) = self.index_type() {
                            f.field("index_type", &index_type);
                        }
                        if let Ok(is_ordered) = self.is_ordered() {
                            f.field("is_ordered", &is_ordered);
                        }
                        if let Ok(dictionary_kind) = self.dictionary_kind() {
                            f.field("dictionary_kind", &dictionary_kind);
                        }
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for DictionaryEncodingRef<'a> {
                    type Value = DictionaryEncoding;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(DictionaryEncoding {
                            id: planus::ToOwned::to_owned(self.id()?)?,
                            index_type: if let Some(index_type) = self.index_type()? {
                                Some(Box::new(planus::ToOwned::to_owned(index_type)?))
                            } else {
                                None
                            },
                            is_ordered: planus::ToOwned::to_owned(self.is_ordered()?)?,
                            dictionary_kind: planus::ToOwned::to_owned(self.dictionary_kind()?)?,
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for DictionaryEncodingRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for DictionaryEncoding {
                    type Output = planus::Result<DictionaryEncodingRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[DictionaryEncodingRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<DictionaryEncoding>> for DictionaryEncoding {
                    type Value = planus::Offset<DictionaryEncoding>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<DictionaryEncoding>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for DictionaryEncodingRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[DictionaryEncodingRef]",
                                "read_as_root",
                                0,
                            )
                        })
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct Field {
                    pub name: Option<String>,
                    pub nullable: bool,
                    pub type_: Option<self::Type>,
                    pub dictionary: Option<Box<self::DictionaryEncoding>>,
                    pub children: Option<Vec<self::Field>>,
                    pub custom_metadata: Option<Vec<self::KeyValue>>,
                }

                impl Field {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(
                        builder: &mut planus::Builder,
                        name: impl planus::WriteAsOptional<planus::Offset<str>>,
                        nullable: impl planus::WriteAsDefault<bool, bool>,
                        type_: impl planus::WriteAsOptionalUnion<self::Type>,
                        dictionary: impl planus::WriteAsOptional<
                            planus::Offset<self::DictionaryEncoding>,
                        >,
                        children: impl planus::WriteAsOptional<
                            planus::Offset<[planus::Offset<self::Field>]>,
                        >,
                        custom_metadata: impl planus::WriteAsOptional<
                            planus::Offset<[planus::Offset<self::KeyValue>]>,
                        >,
                    ) -> planus::Offset<Self> {
                        let prepared_name = name.prepare(builder);

                        let prepared_nullable = nullable.prepare(builder, &false);

                        let prepared_type_ = type_.prepare(builder);

                        let prepared_dictionary = dictionary.prepare(builder);

                        let prepared_children = children.prepare(builder);

                        let prepared_custom_metadata = custom_metadata.prepare(builder);

                        let mut table_writer =
                            planus::table_writer::TableWriter::<16, 22>::new(builder);

                        if prepared_name.is_some() {
                            table_writer.calculate_size::<planus::Offset<str>>(2);
                        }
                        if prepared_nullable.is_some() {
                            table_writer.calculate_size::<bool>(4);
                        }
                        if prepared_type_.is_some() {
                            table_writer.calculate_size::<u8>(6);
                            table_writer.calculate_size::<planus::Offset<self::Type>>(8);
                        }
                        if prepared_dictionary.is_some() {
                            table_writer
                                .calculate_size::<planus::Offset<self::DictionaryEncoding>>(10);
                        }
                        if prepared_children.is_some() {
                            table_writer
                                .calculate_size::<planus::Offset<[planus::Offset<self::Field>]>>(
                                    12,
                                );
                        }
                        if prepared_custom_metadata.is_some() {
                            table_writer
                                .calculate_size::<planus::Offset<[planus::Offset<self::KeyValue>]>>(
                                    14,
                                );
                        }

                        table_writer.finish_calculating();

                        unsafe {
                            if let Some(prepared_name) = prepared_name {
                                table_writer.write::<_, _, 4>(0, &prepared_name);
                            }
                            if let Some(prepared_type_) = prepared_type_ {
                                table_writer.write::<_, _, 4>(3, &prepared_type_.offset);
                            }
                            if let Some(prepared_dictionary) = prepared_dictionary {
                                table_writer.write::<_, _, 4>(4, &prepared_dictionary);
                            }
                            if let Some(prepared_children) = prepared_children {
                                table_writer.write::<_, _, 4>(5, &prepared_children);
                            }
                            if let Some(prepared_custom_metadata) = prepared_custom_metadata {
                                table_writer.write::<_, _, 4>(6, &prepared_custom_metadata);
                            }
                            if let Some(prepared_nullable) = prepared_nullable {
                                table_writer.write::<_, _, 1>(1, &prepared_nullable);
                            }
                            if let Some(prepared_type_) = prepared_type_ {
                                table_writer.write::<_, _, 1>(2, &prepared_type_.tag);
                            }
                        }

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<Field>> for Field {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Field> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<Field>> for Field {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<Field>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<Field> for Field {
                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Field> {
                        Field::create(
                            builder,
                            &self.name,
                            &self.nullable,
                            &self.type_,
                            &self.dictionary,
                            &self.children,
                            &self.custom_metadata,
                        )
                    }
                }

                #[derive(Copy, Clone)]
                pub struct FieldRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> FieldRef<'a> {
                    pub fn name(&self) -> planus::Result<Option<&'a str>> {
                        self.0.access(0, "Field", "name")
                    }

                    pub fn nullable(&self) -> planus::Result<bool> {
                        Ok(self.0.access(1, "Field", "nullable")?.unwrap_or(false))
                    }

                    pub fn type_(&self) -> planus::Result<Option<self::TypeRef<'a>>> {
                        self.0.access_union(2, "Field", "type_")
                    }

                    pub fn dictionary(
                        &self,
                    ) -> planus::Result<Option<self::DictionaryEncodingRef<'a>>>
                    {
                        self.0.access(4, "Field", "dictionary")
                    }

                    pub fn children(
                        &self,
                    ) -> planus::Result<Option<planus::Vector<'a, self::Field>>>
                    {
                        self.0.access(5, "Field", "children")
                    }

                    pub fn custom_metadata(
                        &self,
                    ) -> planus::Result<Option<planus::Vector<'a, self::KeyValue>>>
                    {
                        self.0.access(6, "Field", "custom_metadata")
                    }
                }

                impl<'a> std::fmt::Debug for FieldRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("FieldRef");
                        if let Ok(Some(name)) = self.name() {
                            f.field("name", &name);
                        }
                        if let Ok(nullable) = self.nullable() {
                            f.field("nullable", &nullable);
                        }
                        if let Ok(Some(type_)) = self.type_() {
                            f.field("type_", &type_);
                        }
                        if let Ok(Some(dictionary)) = self.dictionary() {
                            f.field("dictionary", &dictionary);
                        }
                        if let Ok(Some(children)) = self.children() {
                            f.field("children", &children);
                        }
                        if let Ok(Some(custom_metadata)) = self.custom_metadata() {
                            f.field("custom_metadata", &custom_metadata);
                        }
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for FieldRef<'a> {
                    type Value = Field;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(Field {
                            name: if let Some(name) = self.name()? {
                                Some(planus::ToOwned::to_owned(name)?)
                            } else {
                                None
                            },
                            nullable: planus::ToOwned::to_owned(self.nullable()?)?,
                            type_: if let Some(type_) = self.type_()? {
                                Some(planus::ToOwned::to_owned(type_)?)
                            } else {
                                None
                            },
                            dictionary: if let Some(dictionary) = self.dictionary()? {
                                Some(Box::new(planus::ToOwned::to_owned(dictionary)?))
                            } else {
                                None
                            },
                            children: if let Some(children) = self.children()? {
                                Some(planus::ToOwned::to_owned(children)?)
                            } else {
                                None
                            },
                            custom_metadata: if let Some(custom_metadata) =
                                self.custom_metadata()?
                            {
                                Some(planus::ToOwned::to_owned(custom_metadata)?)
                            } else {
                                None
                            },
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for FieldRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for Field {
                    type Output = planus::Result<FieldRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[FieldRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<Field>> for Field {
                    type Value = planus::Offset<Field>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<Field>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for FieldRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location("[FieldRef]", "read_as_root", 0)
                        })
                    }
                }

                #[derive(
                    Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize,
                )]
                #[repr(i16)]
                pub enum Endianness {
                    Little = 0,
                    Big = 1,
                }

                impl TryFrom<i16> for Endianness {
                    type Error = planus::errors::UnknownEnumTagKind;
                    fn try_from(
                        value: i16,
                    ) -> std::result::Result<Self, planus::errors::UnknownEnumTagKind>
                    {
                        #[allow(clippy::match_single_binding)]
                        match value {
                            0 => Ok(Endianness::Little),
                            1 => Ok(Endianness::Big),

                            _ => Err(planus::errors::UnknownEnumTagKind { tag: value as i128 }),
                        }
                    }
                }

                impl From<Endianness> for i16 {
                    fn from(value: Endianness) -> Self {
                        value as i16
                    }
                }

                impl planus::ToOwned for Endianness {
                    type Value = Endianness;
                    #[inline]
                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(self)
                    }
                }

                impl planus::Primitive for Endianness {
                    const ALIGNMENT: usize = 2;
                    const SIZE: usize = 2;
                }

                impl planus::WriteAsPrimitive<Endianness> for Endianness {
                    #[inline]
                    fn write<const N: usize>(
                        &self,
                        cursor: planus::Cursor<'_, N>,
                        buffer_position: u32,
                    ) {
                        (*self as i16).write(cursor, buffer_position);
                    }
                }

                impl planus::WriteAs<Endianness> for Endianness {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(&self, _builder: &mut planus::Builder) -> Endianness {
                        *self
                    }
                }

                impl planus::WriteAsDefault<Endianness, Endianness> for Endianness {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(
                        &self,
                        _builder: &mut planus::Builder,
                        default: &Endianness,
                    ) -> Option<Endianness> {
                        if self == default {
                            None
                        } else {
                            Some(*self)
                        }
                    }
                }

                impl planus::WriteAsOptional<Endianness> for Endianness {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(&self, _builder: &mut planus::Builder) -> Option<Endianness> {
                        Some(*self)
                    }
                }

                impl<'buf> planus::TableRead<'buf> for Endianness {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'buf>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        let n: i16 = planus::TableRead::from_buffer(buffer, offset)?;
                        Ok(n.try_into()?)
                    }
                }

                impl<'buf> planus::VectorRead<'buf> for Endianness {
                    type Output = std::result::Result<Self, planus::errors::UnknownEnumTag>;

                    const STRIDE: usize = 2;
                    #[inline]
                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'buf>,
                        offset: usize,
                    ) -> Self::Output {
                        let value = <i16 as planus::VectorRead>::from_buffer(buffer, offset);
                        let value: std::result::Result<Self, _> = value.try_into();
                        value.map_err(|error_kind| {
                            error_kind.with_error_location(
                                "Endianness",
                                "VectorRead::from_buffer",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl<'buf> planus::VectorWrite<Endianness> for Endianness {
                    const STRIDE: usize = 2;

                    type Value = Self;

                    fn prepare(&self, _builder: &mut planus::Builder) -> Self::Value {
                        *self
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[Self],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 2];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (2 * i) as u32,
                            );
                        }
                    }
                }

                #[derive(Copy, Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
                pub struct Buffer {
                    pub offset: i64,
                    pub length: i64,
                }

                impl planus::Primitive for Buffer {
                    const ALIGNMENT: usize = 8;
                    const SIZE: usize = 16;
                }

                #[allow(clippy::identity_op)]
                impl planus::WriteAsPrimitive<Buffer> for Buffer {
                    fn write<const N: usize>(
                        &self,
                        cursor: planus::Cursor<'_, N>,
                        buffer_position: u32,
                    ) {
                        let (cur, cursor) = cursor.split::<8, 8>();
                        self.offset.write(cur, buffer_position - 0);
                        let (cur, cursor) = cursor.split::<8, 0>();
                        self.length.write(cur, buffer_position - 8);
                        cursor.finish([]);
                    }
                }

                impl planus::WriteAs<Buffer> for Buffer {
                    type Prepared = Self;
                    fn prepare(&self, _builder: &mut planus::Builder) -> Self {
                        *self
                    }
                }

                impl planus::WriteAsOptional<Buffer> for Buffer {
                    type Prepared = Self;
                    fn prepare(&self, _builder: &mut planus::Builder) -> Option<Self> {
                        Some(*self)
                    }
                }

                #[derive(Copy, Clone)]
                pub struct BufferRef<'a>(planus::ArrayWithStartOffset<'a, 16>);

                impl<'a> BufferRef<'a> {
                    pub fn offset(&self) -> i64 {
                        let buffer = self.0.advance_as_array::<8>(0).unwrap();

                        i64::from_le_bytes(*buffer.as_array())
                    }

                    pub fn length(&self) -> i64 {
                        let buffer = self.0.advance_as_array::<8>(8).unwrap();

                        i64::from_le_bytes(*buffer.as_array())
                    }
                }

                impl<'a> std::fmt::Debug for BufferRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("BufferRef");
                        f.field("offset", &self.offset());
                        f.field("length", &self.length());
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for BufferRef<'a> {
                    type Value = Buffer;
                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(Buffer {
                            offset: self.offset(),
                            length: self.length(),
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for BufferRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        let buffer = buffer.advance_as_array::<16>(offset)?;
                        Ok(Self(buffer))
                    }
                }

                impl<'a> planus::VectorRead<'a> for Buffer {
                    const STRIDE: usize = 16;

                    type Output = BufferRef<'a>;
                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> BufferRef<'a> {
                        BufferRef(buffer.unchecked_advance_as_array(offset))
                    }
                }

                impl planus::VectorWrite<Buffer> for Buffer {
                    const STRIDE: usize = 16;

                    type Value = Buffer;

                    fn prepare(&self, _builder: &mut planus::Builder) -> Self::Value {
                        *self
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[Buffer],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 16];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (16 * i) as u32,
                            );
                        }
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct Schema {
                    pub endianness: self::Endianness,
                    pub fields: Option<Vec<self::Field>>,
                    pub custom_metadata: Option<Vec<self::KeyValue>>,
                    pub features: Option<Vec<self::Feature>>,
                }

                impl Schema {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(
                        builder: &mut planus::Builder,
                        endianness: impl planus::WriteAsDefault<self::Endianness, self::Endianness>,
                        fields: impl planus::WriteAsOptional<
                            planus::Offset<[planus::Offset<self::Field>]>,
                        >,
                        custom_metadata: impl planus::WriteAsOptional<
                            planus::Offset<[planus::Offset<self::KeyValue>]>,
                        >,
                        features: impl planus::WriteAsOptional<planus::Offset<[self::Feature]>>,
                    ) -> planus::Offset<Self> {
                        let prepared_endianness =
                            endianness.prepare(builder, &self::Endianness::Little);

                        let prepared_fields = fields.prepare(builder);

                        let prepared_custom_metadata = custom_metadata.prepare(builder);

                        let prepared_features = features.prepare(builder);

                        let mut table_writer =
                            planus::table_writer::TableWriter::<10, 14>::new(builder);

                        if prepared_endianness.is_some() {
                            table_writer.calculate_size::<self::Endianness>(2);
                        }
                        if prepared_fields.is_some() {
                            table_writer
                                .calculate_size::<planus::Offset<[planus::Offset<self::Field>]>>(4);
                        }
                        if prepared_custom_metadata.is_some() {
                            table_writer
                                .calculate_size::<planus::Offset<[planus::Offset<self::KeyValue>]>>(
                                    6,
                                );
                        }
                        if prepared_features.is_some() {
                            table_writer.calculate_size::<planus::Offset<[self::Feature]>>(8);
                        }

                        table_writer.finish_calculating();

                        unsafe {
                            if let Some(prepared_fields) = prepared_fields {
                                table_writer.write::<_, _, 4>(1, &prepared_fields);
                            }
                            if let Some(prepared_custom_metadata) = prepared_custom_metadata {
                                table_writer.write::<_, _, 4>(2, &prepared_custom_metadata);
                            }
                            if let Some(prepared_features) = prepared_features {
                                table_writer.write::<_, _, 4>(3, &prepared_features);
                            }
                            if let Some(prepared_endianness) = prepared_endianness {
                                table_writer.write::<_, _, 2>(0, &prepared_endianness);
                            }
                        }

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<Schema>> for Schema {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Schema> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<Schema>> for Schema {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<Schema>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<Schema> for Schema {
                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Schema> {
                        Schema::create(
                            builder,
                            &self.endianness,
                            &self.fields,
                            &self.custom_metadata,
                            &self.features,
                        )
                    }
                }

                #[derive(Copy, Clone)]
                pub struct SchemaRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> SchemaRef<'a> {
                    pub fn endianness(&self) -> planus::Result<self::Endianness> {
                        Ok(self
                            .0
                            .access(0, "Schema", "endianness")?
                            .unwrap_or(self::Endianness::Little))
                    }

                    pub fn fields(
                        &self,
                    ) -> planus::Result<Option<planus::Vector<'a, self::Field>>>
                    {
                        self.0.access(1, "Schema", "fields")
                    }

                    pub fn custom_metadata(
                        &self,
                    ) -> planus::Result<Option<planus::Vector<'a, self::KeyValue>>>
                    {
                        self.0.access(2, "Schema", "custom_metadata")
                    }

                    pub fn features(
                        &self,
                    ) -> planus::Result<Option<planus::Vector<'a, self::Feature>>>
                    {
                        self.0.access(3, "Schema", "features")
                    }
                }

                impl<'a> std::fmt::Debug for SchemaRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("SchemaRef");
                        if let Ok(endianness) = self.endianness() {
                            f.field("endianness", &endianness);
                        }
                        if let Ok(Some(fields)) = self.fields() {
                            f.field("fields", &fields);
                        }
                        if let Ok(Some(custom_metadata)) = self.custom_metadata() {
                            f.field("custom_metadata", &custom_metadata);
                        }
                        if let Ok(Some(features)) = self.features() {
                            f.field("features", &features);
                        }
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for SchemaRef<'a> {
                    type Value = Schema;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(Schema {
                            endianness: planus::ToOwned::to_owned(self.endianness()?)?,
                            fields: if let Some(fields) = self.fields()? {
                                Some(planus::ToOwned::to_owned(fields)?)
                            } else {
                                None
                            },
                            custom_metadata: if let Some(custom_metadata) =
                                self.custom_metadata()?
                            {
                                Some(planus::ToOwned::to_owned(custom_metadata)?)
                            } else {
                                None
                            },
                            features: if let Some(features) = self.features()? {
                                Some(planus::ToOwned::to_owned(features)?)
                            } else {
                                None
                            },
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for SchemaRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for Schema {
                    type Output = planus::Result<SchemaRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[SchemaRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<Schema>> for Schema {
                    type Value = planus::Offset<Schema>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<Schema>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for SchemaRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location("[SchemaRef]", "read_as_root", 0)
                        })
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct SparseTensorIndexCoo {
                    pub indices_type: Box<self::Int>,
                    pub indices_strides: Option<Vec<i64>>,
                    pub indices_buffer: self::Buffer,
                    pub is_canonical: bool,
                }

                impl SparseTensorIndexCoo {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(
                        builder: &mut planus::Builder,
                        indices_type: impl planus::WriteAs<planus::Offset<self::Int>>,
                        indices_strides: impl planus::WriteAsOptional<planus::Offset<[i64]>>,
                        indices_buffer: impl planus::WriteAs<self::Buffer>,
                        is_canonical: impl planus::WriteAsDefault<bool, bool>,
                    ) -> planus::Offset<Self> {
                        let prepared_indices_type = indices_type.prepare(builder);

                        let prepared_indices_strides = indices_strides.prepare(builder);

                        let prepared_indices_buffer = indices_buffer.prepare(builder);

                        let prepared_is_canonical = is_canonical.prepare(builder, &false);

                        let mut table_writer =
                            planus::table_writer::TableWriter::<10, 25>::new(builder);

                        table_writer.calculate_size::<planus::Offset<self::Int>>(2);
                        if prepared_indices_strides.is_some() {
                            table_writer.calculate_size::<planus::Offset<[i64]>>(4);
                        }
                        table_writer.calculate_size::<self::Buffer>(6);
                        if prepared_is_canonical.is_some() {
                            table_writer.calculate_size::<bool>(8);
                        }

                        table_writer.finish_calculating();

                        unsafe {
                            table_writer.write::<_, _, 16>(2, &prepared_indices_buffer);
                            table_writer.write::<_, _, 4>(0, &prepared_indices_type);
                            if let Some(prepared_indices_strides) = prepared_indices_strides {
                                table_writer.write::<_, _, 4>(1, &prepared_indices_strides);
                            }
                            if let Some(prepared_is_canonical) = prepared_is_canonical {
                                table_writer.write::<_, _, 1>(3, &prepared_is_canonical);
                            }
                        }

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<SparseTensorIndexCoo>> for SparseTensorIndexCoo {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> planus::Offset<SparseTensorIndexCoo> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<SparseTensorIndexCoo>> for SparseTensorIndexCoo {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<SparseTensorIndexCoo>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<SparseTensorIndexCoo> for SparseTensorIndexCoo {
                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> planus::Offset<SparseTensorIndexCoo> {
                        SparseTensorIndexCoo::create(
                            builder,
                            &self.indices_type,
                            &self.indices_strides,
                            &self.indices_buffer,
                            &self.is_canonical,
                        )
                    }
                }

                #[derive(Copy, Clone)]
                pub struct SparseTensorIndexCooRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> SparseTensorIndexCooRef<'a> {
                    pub fn indices_type(&self) -> planus::Result<self::IntRef<'a>> {
                        self.0
                            .access_required(0, "SparseTensorIndexCoo", "indices_type")
                    }

                    pub fn indices_strides(
                        &self,
                    ) -> planus::Result<Option<planus::Vector<'a, i64>>> {
                        self.0.access(1, "SparseTensorIndexCoo", "indices_strides")
                    }

                    pub fn indices_buffer(&self) -> planus::Result<self::BufferRef<'a>> {
                        self.0
                            .access_required(2, "SparseTensorIndexCoo", "indices_buffer")
                    }

                    pub fn is_canonical(&self) -> planus::Result<bool> {
                        Ok(self
                            .0
                            .access(3, "SparseTensorIndexCoo", "is_canonical")?
                            .unwrap_or(false))
                    }
                }

                impl<'a> std::fmt::Debug for SparseTensorIndexCooRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("SparseTensorIndexCooRef");
                        if let Ok(indices_type) = self.indices_type() {
                            f.field("indices_type", &indices_type);
                        }
                        if let Ok(Some(indices_strides)) = self.indices_strides() {
                            f.field("indices_strides", &indices_strides);
                        }
                        if let Ok(indices_buffer) = self.indices_buffer() {
                            f.field("indices_buffer", &indices_buffer);
                        }
                        if let Ok(is_canonical) = self.is_canonical() {
                            f.field("is_canonical", &is_canonical);
                        }
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for SparseTensorIndexCooRef<'a> {
                    type Value = SparseTensorIndexCoo;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(SparseTensorIndexCoo {
                            indices_type: Box::new(planus::ToOwned::to_owned(
                                self.indices_type()?,
                            )?),
                            indices_strides: if let Some(indices_strides) =
                                self.indices_strides()?
                            {
                                Some(planus::ToOwned::to_owned(indices_strides)?)
                            } else {
                                None
                            },
                            indices_buffer: planus::ToOwned::to_owned(self.indices_buffer()?)?,
                            is_canonical: planus::ToOwned::to_owned(self.is_canonical()?)?,
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for SparseTensorIndexCooRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for SparseTensorIndexCoo {
                    type Output = planus::Result<SparseTensorIndexCooRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[SparseTensorIndexCooRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<SparseTensorIndexCoo>> for SparseTensorIndexCoo {
                    type Value = planus::Offset<SparseTensorIndexCoo>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<SparseTensorIndexCoo>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for SparseTensorIndexCooRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[SparseTensorIndexCooRef]",
                                "read_as_root",
                                0,
                            )
                        })
                    }
                }

                #[derive(
                    Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize,
                )]
                #[repr(i16)]
                pub enum SparseMatrixCompressedAxis {
                    Row = 0,
                    Column = 1,
                }

                impl TryFrom<i16> for SparseMatrixCompressedAxis {
                    type Error = planus::errors::UnknownEnumTagKind;
                    fn try_from(
                        value: i16,
                    ) -> std::result::Result<Self, planus::errors::UnknownEnumTagKind>
                    {
                        #[allow(clippy::match_single_binding)]
                        match value {
                            0 => Ok(SparseMatrixCompressedAxis::Row),
                            1 => Ok(SparseMatrixCompressedAxis::Column),

                            _ => Err(planus::errors::UnknownEnumTagKind { tag: value as i128 }),
                        }
                    }
                }

                impl From<SparseMatrixCompressedAxis> for i16 {
                    fn from(value: SparseMatrixCompressedAxis) -> Self {
                        value as i16
                    }
                }

                impl planus::ToOwned for SparseMatrixCompressedAxis {
                    type Value = SparseMatrixCompressedAxis;
                    #[inline]
                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(self)
                    }
                }

                impl planus::Primitive for SparseMatrixCompressedAxis {
                    const ALIGNMENT: usize = 2;
                    const SIZE: usize = 2;
                }

                impl planus::WriteAsPrimitive<SparseMatrixCompressedAxis> for SparseMatrixCompressedAxis {
                    #[inline]
                    fn write<const N: usize>(
                        &self,
                        cursor: planus::Cursor<'_, N>,
                        buffer_position: u32,
                    ) {
                        (*self as i16).write(cursor, buffer_position);
                    }
                }

                impl planus::WriteAs<SparseMatrixCompressedAxis> for SparseMatrixCompressedAxis {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(
                        &self,
                        _builder: &mut planus::Builder,
                    ) -> SparseMatrixCompressedAxis {
                        *self
                    }
                }

                impl planus::WriteAsDefault<SparseMatrixCompressedAxis, SparseMatrixCompressedAxis>
                    for SparseMatrixCompressedAxis
                {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(
                        &self,
                        _builder: &mut planus::Builder,
                        default: &SparseMatrixCompressedAxis,
                    ) -> Option<SparseMatrixCompressedAxis> {
                        if self == default {
                            None
                        } else {
                            Some(*self)
                        }
                    }
                }

                impl planus::WriteAsOptional<SparseMatrixCompressedAxis> for SparseMatrixCompressedAxis {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(
                        &self,
                        _builder: &mut planus::Builder,
                    ) -> Option<SparseMatrixCompressedAxis> {
                        Some(*self)
                    }
                }

                impl<'buf> planus::TableRead<'buf> for SparseMatrixCompressedAxis {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'buf>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        let n: i16 = planus::TableRead::from_buffer(buffer, offset)?;
                        Ok(n.try_into()?)
                    }
                }

                impl<'buf> planus::VectorRead<'buf> for SparseMatrixCompressedAxis {
                    type Output = std::result::Result<Self, planus::errors::UnknownEnumTag>;

                    const STRIDE: usize = 2;
                    #[inline]
                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'buf>,
                        offset: usize,
                    ) -> Self::Output {
                        let value = <i16 as planus::VectorRead>::from_buffer(buffer, offset);
                        let value: std::result::Result<Self, _> = value.try_into();
                        value.map_err(|error_kind| {
                            error_kind.with_error_location(
                                "SparseMatrixCompressedAxis",
                                "VectorRead::from_buffer",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl<'buf> planus::VectorWrite<SparseMatrixCompressedAxis> for SparseMatrixCompressedAxis {
                    const STRIDE: usize = 2;

                    type Value = Self;

                    fn prepare(&self, _builder: &mut planus::Builder) -> Self::Value {
                        *self
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[Self],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 2];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (2 * i) as u32,
                            );
                        }
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct SparseMatrixIndexCsx {
                    pub compressed_axis: self::SparseMatrixCompressedAxis,
                    pub indptr_type: Box<self::Int>,
                    pub indptr_buffer: self::Buffer,
                    pub indices_type: Box<self::Int>,
                    pub indices_buffer: self::Buffer,
                }

                impl SparseMatrixIndexCsx {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(
                        builder: &mut planus::Builder,
                        compressed_axis: impl planus::WriteAsDefault<
                            self::SparseMatrixCompressedAxis,
                            self::SparseMatrixCompressedAxis,
                        >,
                        indptr_type: impl planus::WriteAs<planus::Offset<self::Int>>,
                        indptr_buffer: impl planus::WriteAs<self::Buffer>,
                        indices_type: impl planus::WriteAs<planus::Offset<self::Int>>,
                        indices_buffer: impl planus::WriteAs<self::Buffer>,
                    ) -> planus::Offset<Self> {
                        let prepared_compressed_axis = compressed_axis
                            .prepare(builder, &self::SparseMatrixCompressedAxis::Row);

                        let prepared_indptr_type = indptr_type.prepare(builder);

                        let prepared_indptr_buffer = indptr_buffer.prepare(builder);

                        let prepared_indices_type = indices_type.prepare(builder);

                        let prepared_indices_buffer = indices_buffer.prepare(builder);

                        let mut table_writer =
                            planus::table_writer::TableWriter::<12, 42>::new(builder);

                        if prepared_compressed_axis.is_some() {
                            table_writer.calculate_size::<self::SparseMatrixCompressedAxis>(2);
                        }
                        table_writer.calculate_size::<planus::Offset<self::Int>>(4);
                        table_writer.calculate_size::<self::Buffer>(6);
                        table_writer.calculate_size::<planus::Offset<self::Int>>(8);
                        table_writer.calculate_size::<self::Buffer>(10);

                        table_writer.finish_calculating();

                        unsafe {
                            table_writer.write::<_, _, 16>(2, &prepared_indptr_buffer);
                            table_writer.write::<_, _, 16>(4, &prepared_indices_buffer);
                            table_writer.write::<_, _, 4>(1, &prepared_indptr_type);
                            table_writer.write::<_, _, 4>(3, &prepared_indices_type);
                            if let Some(prepared_compressed_axis) = prepared_compressed_axis {
                                table_writer.write::<_, _, 2>(0, &prepared_compressed_axis);
                            }
                        }

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<SparseMatrixIndexCsx>> for SparseMatrixIndexCsx {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> planus::Offset<SparseMatrixIndexCsx> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<SparseMatrixIndexCsx>> for SparseMatrixIndexCsx {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<SparseMatrixIndexCsx>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<SparseMatrixIndexCsx> for SparseMatrixIndexCsx {
                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> planus::Offset<SparseMatrixIndexCsx> {
                        SparseMatrixIndexCsx::create(
                            builder,
                            &self.compressed_axis,
                            &self.indptr_type,
                            &self.indptr_buffer,
                            &self.indices_type,
                            &self.indices_buffer,
                        )
                    }
                }

                #[derive(Copy, Clone)]
                pub struct SparseMatrixIndexCsxRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> SparseMatrixIndexCsxRef<'a> {
                    pub fn compressed_axis(
                        &self,
                    ) -> planus::Result<self::SparseMatrixCompressedAxis> {
                        Ok(self
                            .0
                            .access(0, "SparseMatrixIndexCsx", "compressed_axis")?
                            .unwrap_or(self::SparseMatrixCompressedAxis::Row))
                    }

                    pub fn indptr_type(&self) -> planus::Result<self::IntRef<'a>> {
                        self.0
                            .access_required(1, "SparseMatrixIndexCsx", "indptr_type")
                    }

                    pub fn indptr_buffer(&self) -> planus::Result<self::BufferRef<'a>> {
                        self.0
                            .access_required(2, "SparseMatrixIndexCsx", "indptr_buffer")
                    }

                    pub fn indices_type(&self) -> planus::Result<self::IntRef<'a>> {
                        self.0
                            .access_required(3, "SparseMatrixIndexCsx", "indices_type")
                    }

                    pub fn indices_buffer(&self) -> planus::Result<self::BufferRef<'a>> {
                        self.0
                            .access_required(4, "SparseMatrixIndexCsx", "indices_buffer")
                    }
                }

                impl<'a> std::fmt::Debug for SparseMatrixIndexCsxRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("SparseMatrixIndexCsxRef");
                        if let Ok(compressed_axis) = self.compressed_axis() {
                            f.field("compressed_axis", &compressed_axis);
                        }
                        if let Ok(indptr_type) = self.indptr_type() {
                            f.field("indptr_type", &indptr_type);
                        }
                        if let Ok(indptr_buffer) = self.indptr_buffer() {
                            f.field("indptr_buffer", &indptr_buffer);
                        }
                        if let Ok(indices_type) = self.indices_type() {
                            f.field("indices_type", &indices_type);
                        }
                        if let Ok(indices_buffer) = self.indices_buffer() {
                            f.field("indices_buffer", &indices_buffer);
                        }
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for SparseMatrixIndexCsxRef<'a> {
                    type Value = SparseMatrixIndexCsx;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(SparseMatrixIndexCsx {
                            compressed_axis: planus::ToOwned::to_owned(self.compressed_axis()?)?,
                            indptr_type: Box::new(planus::ToOwned::to_owned(self.indptr_type()?)?),
                            indptr_buffer: planus::ToOwned::to_owned(self.indptr_buffer()?)?,
                            indices_type: Box::new(planus::ToOwned::to_owned(
                                self.indices_type()?,
                            )?),
                            indices_buffer: planus::ToOwned::to_owned(self.indices_buffer()?)?,
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for SparseMatrixIndexCsxRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for SparseMatrixIndexCsx {
                    type Output = planus::Result<SparseMatrixIndexCsxRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[SparseMatrixIndexCsxRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<SparseMatrixIndexCsx>> for SparseMatrixIndexCsx {
                    type Value = planus::Offset<SparseMatrixIndexCsx>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<SparseMatrixIndexCsx>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for SparseMatrixIndexCsxRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[SparseMatrixIndexCsxRef]",
                                "read_as_root",
                                0,
                            )
                        })
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct SparseTensorIndexCsf {
                    pub indptr_type: Box<self::Int>,
                    pub indptr_buffers: Vec<self::Buffer>,
                    pub indices_type: Box<self::Int>,
                    pub indices_buffers: Vec<self::Buffer>,
                    pub axis_order: Vec<i32>,
                }

                impl SparseTensorIndexCsf {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(
                        builder: &mut planus::Builder,
                        indptr_type: impl planus::WriteAs<planus::Offset<self::Int>>,
                        indptr_buffers: impl planus::WriteAs<planus::Offset<[self::Buffer]>>,
                        indices_type: impl planus::WriteAs<planus::Offset<self::Int>>,
                        indices_buffers: impl planus::WriteAs<planus::Offset<[self::Buffer]>>,
                        axis_order: impl planus::WriteAs<planus::Offset<[i32]>>,
                    ) -> planus::Offset<Self> {
                        let prepared_indptr_type = indptr_type.prepare(builder);

                        let prepared_indptr_buffers = indptr_buffers.prepare(builder);

                        let prepared_indices_type = indices_type.prepare(builder);

                        let prepared_indices_buffers = indices_buffers.prepare(builder);

                        let prepared_axis_order = axis_order.prepare(builder);

                        let mut table_writer =
                            planus::table_writer::TableWriter::<12, 20>::new(builder);

                        table_writer.calculate_size::<planus::Offset<self::Int>>(2);
                        table_writer.calculate_size::<planus::Offset<[self::Buffer]>>(4);
                        table_writer.calculate_size::<planus::Offset<self::Int>>(6);
                        table_writer.calculate_size::<planus::Offset<[self::Buffer]>>(8);
                        table_writer.calculate_size::<planus::Offset<[i32]>>(10);

                        table_writer.finish_calculating();

                        unsafe {
                            table_writer.write::<_, _, 4>(0, &prepared_indptr_type);
                            table_writer.write::<_, _, 4>(1, &prepared_indptr_buffers);
                            table_writer.write::<_, _, 4>(2, &prepared_indices_type);
                            table_writer.write::<_, _, 4>(3, &prepared_indices_buffers);
                            table_writer.write::<_, _, 4>(4, &prepared_axis_order);
                        }

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<SparseTensorIndexCsf>> for SparseTensorIndexCsf {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> planus::Offset<SparseTensorIndexCsf> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<SparseTensorIndexCsf>> for SparseTensorIndexCsf {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<SparseTensorIndexCsf>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<SparseTensorIndexCsf> for SparseTensorIndexCsf {
                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> planus::Offset<SparseTensorIndexCsf> {
                        SparseTensorIndexCsf::create(
                            builder,
                            &self.indptr_type,
                            &self.indptr_buffers,
                            &self.indices_type,
                            &self.indices_buffers,
                            &self.axis_order,
                        )
                    }
                }

                #[derive(Copy, Clone)]
                pub struct SparseTensorIndexCsfRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> SparseTensorIndexCsfRef<'a> {
                    pub fn indptr_type(&self) -> planus::Result<self::IntRef<'a>> {
                        self.0
                            .access_required(0, "SparseTensorIndexCsf", "indptr_type")
                    }

                    pub fn indptr_buffers(
                        &self,
                    ) -> planus::Result<planus::Vector<'a, self::Buffer>> {
                        self.0
                            .access_required(1, "SparseTensorIndexCsf", "indptr_buffers")
                    }

                    pub fn indices_type(&self) -> planus::Result<self::IntRef<'a>> {
                        self.0
                            .access_required(2, "SparseTensorIndexCsf", "indices_type")
                    }

                    pub fn indices_buffers(
                        &self,
                    ) -> planus::Result<planus::Vector<'a, self::Buffer>> {
                        self.0
                            .access_required(3, "SparseTensorIndexCsf", "indices_buffers")
                    }

                    pub fn axis_order(&self) -> planus::Result<planus::Vector<'a, i32>> {
                        self.0
                            .access_required(4, "SparseTensorIndexCsf", "axis_order")
                    }
                }

                impl<'a> std::fmt::Debug for SparseTensorIndexCsfRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("SparseTensorIndexCsfRef");
                        if let Ok(indptr_type) = self.indptr_type() {
                            f.field("indptr_type", &indptr_type);
                        }
                        if let Ok(indptr_buffers) = self.indptr_buffers() {
                            f.field("indptr_buffers", &indptr_buffers);
                        }
                        if let Ok(indices_type) = self.indices_type() {
                            f.field("indices_type", &indices_type);
                        }
                        if let Ok(indices_buffers) = self.indices_buffers() {
                            f.field("indices_buffers", &indices_buffers);
                        }
                        if let Ok(axis_order) = self.axis_order() {
                            f.field("axis_order", &axis_order);
                        }
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for SparseTensorIndexCsfRef<'a> {
                    type Value = SparseTensorIndexCsf;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(SparseTensorIndexCsf {
                            indptr_type: Box::new(planus::ToOwned::to_owned(self.indptr_type()?)?),
                            indptr_buffers: planus::ToOwned::to_owned(self.indptr_buffers()?)?,
                            indices_type: Box::new(planus::ToOwned::to_owned(
                                self.indices_type()?,
                            )?),
                            indices_buffers: planus::ToOwned::to_owned(self.indices_buffers()?)?,
                            axis_order: planus::ToOwned::to_owned(self.axis_order()?)?,
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for SparseTensorIndexCsfRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for SparseTensorIndexCsf {
                    type Output = planus::Result<SparseTensorIndexCsfRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[SparseTensorIndexCsfRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<SparseTensorIndexCsf>> for SparseTensorIndexCsf {
                    type Value = planus::Offset<SparseTensorIndexCsf>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<SparseTensorIndexCsf>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for SparseTensorIndexCsfRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[SparseTensorIndexCsfRef]",
                                "read_as_root",
                                0,
                            )
                        })
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub enum SparseTensorIndex {
                    SparseTensorIndexCoo(Box<self::SparseTensorIndexCoo>),
                    SparseMatrixIndexCsx(Box<self::SparseMatrixIndexCsx>),
                    SparseTensorIndexCsf(Box<self::SparseTensorIndexCsf>),
                }

                impl SparseTensorIndex {
                    pub fn create_sparse_tensor_index_coo(
                        builder: &mut planus::Builder,
                        value: impl planus::WriteAsOffset<self::SparseTensorIndexCoo>,
                    ) -> planus::UnionOffset<Self> {
                        planus::UnionOffset::new(1, value.prepare(builder).downcast())
                    }

                    pub fn create_sparse_matrix_index_csx(
                        builder: &mut planus::Builder,
                        value: impl planus::WriteAsOffset<self::SparseMatrixIndexCsx>,
                    ) -> planus::UnionOffset<Self> {
                        planus::UnionOffset::new(2, value.prepare(builder).downcast())
                    }

                    pub fn create_sparse_tensor_index_csf(
                        builder: &mut planus::Builder,
                        value: impl planus::WriteAsOffset<self::SparseTensorIndexCsf>,
                    ) -> planus::UnionOffset<Self> {
                        planus::UnionOffset::new(3, value.prepare(builder).downcast())
                    }
                }

                impl planus::WriteAsUnion<SparseTensorIndex> for SparseTensorIndex {
                    fn prepare(&self, builder: &mut planus::Builder) -> planus::UnionOffset<Self> {
                        match self {
                            Self::SparseTensorIndexCoo(value) => {
                                Self::create_sparse_tensor_index_coo(builder, value)
                            }
                            Self::SparseMatrixIndexCsx(value) => {
                                Self::create_sparse_matrix_index_csx(builder, value)
                            }
                            Self::SparseTensorIndexCsf(value) => {
                                Self::create_sparse_tensor_index_csf(builder, value)
                            }
                        }
                    }
                }

                impl planus::WriteAsOptionalUnion<SparseTensorIndex> for SparseTensorIndex {
                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::UnionOffset<Self>> {
                        Some(planus::WriteAsUnion::prepare(self, builder))
                    }
                }

                #[derive(Copy, Clone, Debug)]
                pub enum SparseTensorIndexRef<'a> {
                    SparseTensorIndexCoo(self::SparseTensorIndexCooRef<'a>),
                    SparseMatrixIndexCsx(self::SparseMatrixIndexCsxRef<'a>),
                    SparseTensorIndexCsf(self::SparseTensorIndexCsfRef<'a>),
                }

                impl<'a> planus::ToOwned for SparseTensorIndexRef<'a> {
                    type Value = SparseTensorIndex;

                    fn to_owned(self) -> planus::Result<SparseTensorIndex> {
                        Ok(match self {
                            Self::SparseTensorIndexCoo(value) => {
                                SparseTensorIndex::SparseTensorIndexCoo(Box::new(
                                    planus::ToOwned::to_owned(value)?,
                                ))
                            }

                            Self::SparseMatrixIndexCsx(value) => {
                                SparseTensorIndex::SparseMatrixIndexCsx(Box::new(
                                    planus::ToOwned::to_owned(value)?,
                                ))
                            }

                            Self::SparseTensorIndexCsf(value) => {
                                SparseTensorIndex::SparseTensorIndexCsf(Box::new(
                                    planus::ToOwned::to_owned(value)?,
                                ))
                            }
                        })
                    }
                }

                impl<'a> planus::TableReadUnion<'a> for SparseTensorIndexRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        field_offset: usize,
                        tag: u8,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        match tag {
                            1 => Ok(Self::SparseTensorIndexCoo(planus::TableRead::from_buffer(
                                buffer,
                                field_offset,
                            )?)),
                            2 => Ok(Self::SparseMatrixIndexCsx(planus::TableRead::from_buffer(
                                buffer,
                                field_offset,
                            )?)),
                            3 => Ok(Self::SparseTensorIndexCsf(planus::TableRead::from_buffer(
                                buffer,
                                field_offset,
                            )?)),
                            _ => Err(planus::errors::ErrorKind::UnknownUnionTag { tag }),
                        }
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct SparseTensor {
                    pub type_: self::Type,
                    pub shape: Vec<self::TensorDim>,
                    pub non_zero_length: i64,
                    pub sparse_index: self::SparseTensorIndex,
                    pub data: self::Buffer,
                }

                impl SparseTensor {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(
                        builder: &mut planus::Builder,
                        type_: impl planus::WriteAsUnion<self::Type>,
                        shape: impl planus::WriteAs<planus::Offset<[planus::Offset<self::TensorDim>]>>,
                        non_zero_length: impl planus::WriteAsDefault<i64, i64>,
                        sparse_index: impl planus::WriteAsUnion<self::SparseTensorIndex>,
                        data: impl planus::WriteAs<self::Buffer>,
                    ) -> planus::Offset<Self> {
                        let prepared_type_ = type_.prepare(builder);

                        let prepared_shape = shape.prepare(builder);

                        let prepared_non_zero_length = non_zero_length.prepare(builder, &0);

                        let prepared_sparse_index = sparse_index.prepare(builder);

                        let prepared_data = data.prepare(builder);

                        let mut table_writer =
                            planus::table_writer::TableWriter::<16, 38>::new(builder);

                        table_writer.calculate_size::<u8>(2);
                        table_writer.calculate_size::<planus::Offset<self::Type>>(4);
                        table_writer
                            .calculate_size::<planus::Offset<[planus::Offset<self::TensorDim>]>>(6);
                        if prepared_non_zero_length.is_some() {
                            table_writer.calculate_size::<i64>(8);
                        }
                        table_writer.calculate_size::<u8>(10);
                        table_writer.calculate_size::<planus::Offset<self::SparseTensorIndex>>(12);
                        table_writer.calculate_size::<self::Buffer>(14);

                        table_writer.finish_calculating();

                        unsafe {
                            if let Some(prepared_non_zero_length) = prepared_non_zero_length {
                                table_writer.write::<_, _, 8>(3, &prepared_non_zero_length);
                            }
                            table_writer.write::<_, _, 16>(6, &prepared_data);
                            table_writer.write::<_, _, 4>(1, &prepared_type_.offset);
                            table_writer.write::<_, _, 4>(2, &prepared_shape);
                            table_writer.write::<_, _, 4>(5, &prepared_sparse_index.offset);
                            table_writer.write::<_, _, 1>(0, &prepared_type_.tag);
                            table_writer.write::<_, _, 1>(4, &prepared_sparse_index.tag);
                        }

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<SparseTensor>> for SparseTensor {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> planus::Offset<SparseTensor> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<SparseTensor>> for SparseTensor {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<SparseTensor>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<SparseTensor> for SparseTensor {
                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> planus::Offset<SparseTensor> {
                        SparseTensor::create(
                            builder,
                            &self.type_,
                            &self.shape,
                            &self.non_zero_length,
                            &self.sparse_index,
                            &self.data,
                        )
                    }
                }

                #[derive(Copy, Clone)]
                pub struct SparseTensorRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> SparseTensorRef<'a> {
                    pub fn type_(&self) -> planus::Result<self::TypeRef<'a>> {
                        self.0.access_union_required(0, "SparseTensor", "type_")
                    }

                    pub fn shape(&self) -> planus::Result<planus::Vector<'a, self::TensorDim>> {
                        self.0.access_required(2, "SparseTensor", "shape")
                    }

                    pub fn non_zero_length(&self) -> planus::Result<i64> {
                        Ok(self
                            .0
                            .access(3, "SparseTensor", "non_zero_length")?
                            .unwrap_or(0))
                    }

                    pub fn sparse_index(&self) -> planus::Result<self::SparseTensorIndexRef<'a>> {
                        self.0
                            .access_union_required(4, "SparseTensor", "sparse_index")
                    }

                    pub fn data(&self) -> planus::Result<self::BufferRef<'a>> {
                        self.0.access_required(6, "SparseTensor", "data")
                    }
                }

                impl<'a> std::fmt::Debug for SparseTensorRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("SparseTensorRef");
                        if let Ok(type_) = self.type_() {
                            f.field("type_", &type_);
                        }
                        if let Ok(shape) = self.shape() {
                            f.field("shape", &shape);
                        }
                        if let Ok(non_zero_length) = self.non_zero_length() {
                            f.field("non_zero_length", &non_zero_length);
                        }
                        if let Ok(sparse_index) = self.sparse_index() {
                            f.field("sparse_index", &sparse_index);
                        }
                        if let Ok(data) = self.data() {
                            f.field("data", &data);
                        }
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for SparseTensorRef<'a> {
                    type Value = SparseTensor;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(SparseTensor {
                            type_: planus::ToOwned::to_owned(self.type_()?)?,
                            shape: planus::ToOwned::to_owned(self.shape()?)?,
                            non_zero_length: planus::ToOwned::to_owned(self.non_zero_length()?)?,
                            sparse_index: planus::ToOwned::to_owned(self.sparse_index()?)?,
                            data: planus::ToOwned::to_owned(self.data()?)?,
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for SparseTensorRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for SparseTensor {
                    type Output = planus::Result<SparseTensorRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[SparseTensorRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<SparseTensor>> for SparseTensor {
                    type Value = planus::Offset<SparseTensor>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<SparseTensor>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for SparseTensorRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location("[SparseTensorRef]", "read_as_root", 0)
                        })
                    }
                }

                #[derive(Copy, Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
                pub struct FieldNode {
                    pub length: i64,
                    pub null_count: i64,
                }

                impl planus::Primitive for FieldNode {
                    const ALIGNMENT: usize = 8;
                    const SIZE: usize = 16;
                }

                #[allow(clippy::identity_op)]
                impl planus::WriteAsPrimitive<FieldNode> for FieldNode {
                    fn write<const N: usize>(
                        &self,
                        cursor: planus::Cursor<'_, N>,
                        buffer_position: u32,
                    ) {
                        let (cur, cursor) = cursor.split::<8, 8>();
                        self.length.write(cur, buffer_position - 0);
                        let (cur, cursor) = cursor.split::<8, 0>();
                        self.null_count.write(cur, buffer_position - 8);
                        cursor.finish([]);
                    }
                }

                impl planus::WriteAs<FieldNode> for FieldNode {
                    type Prepared = Self;
                    fn prepare(&self, _builder: &mut planus::Builder) -> Self {
                        *self
                    }
                }

                impl planus::WriteAsOptional<FieldNode> for FieldNode {
                    type Prepared = Self;
                    fn prepare(&self, _builder: &mut planus::Builder) -> Option<Self> {
                        Some(*self)
                    }
                }

                #[derive(Copy, Clone)]
                pub struct FieldNodeRef<'a>(planus::ArrayWithStartOffset<'a, 16>);

                impl<'a> FieldNodeRef<'a> {
                    pub fn length(&self) -> i64 {
                        let buffer = self.0.advance_as_array::<8>(0).unwrap();

                        i64::from_le_bytes(*buffer.as_array())
                    }

                    pub fn null_count(&self) -> i64 {
                        let buffer = self.0.advance_as_array::<8>(8).unwrap();

                        i64::from_le_bytes(*buffer.as_array())
                    }
                }

                impl<'a> std::fmt::Debug for FieldNodeRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("FieldNodeRef");
                        f.field("length", &self.length());
                        f.field("null_count", &self.null_count());
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for FieldNodeRef<'a> {
                    type Value = FieldNode;
                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(FieldNode {
                            length: self.length(),
                            null_count: self.null_count(),
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for FieldNodeRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        let buffer = buffer.advance_as_array::<16>(offset)?;
                        Ok(Self(buffer))
                    }
                }

                impl<'a> planus::VectorRead<'a> for FieldNode {
                    const STRIDE: usize = 16;

                    type Output = FieldNodeRef<'a>;
                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> FieldNodeRef<'a> {
                        FieldNodeRef(buffer.unchecked_advance_as_array(offset))
                    }
                }

                impl planus::VectorWrite<FieldNode> for FieldNode {
                    const STRIDE: usize = 16;

                    type Value = FieldNode;

                    fn prepare(&self, _builder: &mut planus::Builder) -> Self::Value {
                        *self
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[FieldNode],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 16];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (16 * i) as u32,
                            );
                        }
                    }
                }

                #[derive(
                    Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize,
                )]
                #[repr(i8)]
                pub enum CompressionType {
                    Lz4Frame = 0,
                    Zstd = 1,
                }

                impl TryFrom<i8> for CompressionType {
                    type Error = planus::errors::UnknownEnumTagKind;
                    fn try_from(
                        value: i8,
                    ) -> std::result::Result<Self, planus::errors::UnknownEnumTagKind>
                    {
                        #[allow(clippy::match_single_binding)]
                        match value {
                            0 => Ok(CompressionType::Lz4Frame),
                            1 => Ok(CompressionType::Zstd),

                            _ => Err(planus::errors::UnknownEnumTagKind { tag: value as i128 }),
                        }
                    }
                }

                impl From<CompressionType> for i8 {
                    fn from(value: CompressionType) -> Self {
                        value as i8
                    }
                }

                impl planus::ToOwned for CompressionType {
                    type Value = CompressionType;
                    #[inline]
                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(self)
                    }
                }

                impl planus::Primitive for CompressionType {
                    const ALIGNMENT: usize = 1;
                    const SIZE: usize = 1;
                }

                impl planus::WriteAsPrimitive<CompressionType> for CompressionType {
                    #[inline]
                    fn write<const N: usize>(
                        &self,
                        cursor: planus::Cursor<'_, N>,
                        buffer_position: u32,
                    ) {
                        (*self as i8).write(cursor, buffer_position);
                    }
                }

                impl planus::WriteAs<CompressionType> for CompressionType {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(&self, _builder: &mut planus::Builder) -> CompressionType {
                        *self
                    }
                }

                impl planus::WriteAsDefault<CompressionType, CompressionType> for CompressionType {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(
                        &self,
                        _builder: &mut planus::Builder,
                        default: &CompressionType,
                    ) -> Option<CompressionType> {
                        if self == default {
                            None
                        } else {
                            Some(*self)
                        }
                    }
                }

                impl planus::WriteAsOptional<CompressionType> for CompressionType {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(&self, _builder: &mut planus::Builder) -> Option<CompressionType> {
                        Some(*self)
                    }
                }

                impl<'buf> planus::TableRead<'buf> for CompressionType {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'buf>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        let n: i8 = planus::TableRead::from_buffer(buffer, offset)?;
                        Ok(n.try_into()?)
                    }
                }

                impl<'buf> planus::VectorRead<'buf> for CompressionType {
                    type Output = std::result::Result<Self, planus::errors::UnknownEnumTag>;

                    const STRIDE: usize = 1;
                    #[inline]
                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'buf>,
                        offset: usize,
                    ) -> Self::Output {
                        let value = <i8 as planus::VectorRead>::from_buffer(buffer, offset);
                        let value: std::result::Result<Self, _> = value.try_into();
                        value.map_err(|error_kind| {
                            error_kind.with_error_location(
                                "CompressionType",
                                "VectorRead::from_buffer",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl<'buf> planus::VectorWrite<CompressionType> for CompressionType {
                    const STRIDE: usize = 1;

                    type Value = Self;

                    fn prepare(&self, _builder: &mut planus::Builder) -> Self::Value {
                        *self
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[Self],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 1];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - i as u32,
                            );
                        }
                    }
                }

                #[derive(
                    Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize,
                )]
                #[repr(i8)]
                pub enum BodyCompressionMethod {
                    Buffer = 0,
                }

                impl TryFrom<i8> for BodyCompressionMethod {
                    type Error = planus::errors::UnknownEnumTagKind;
                    fn try_from(
                        value: i8,
                    ) -> std::result::Result<Self, planus::errors::UnknownEnumTagKind>
                    {
                        #[allow(clippy::match_single_binding)]
                        match value {
                            0 => Ok(BodyCompressionMethod::Buffer),

                            _ => Err(planus::errors::UnknownEnumTagKind { tag: value as i128 }),
                        }
                    }
                }

                impl From<BodyCompressionMethod> for i8 {
                    fn from(value: BodyCompressionMethod) -> Self {
                        value as i8
                    }
                }

                impl planus::ToOwned for BodyCompressionMethod {
                    type Value = BodyCompressionMethod;
                    #[inline]
                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(self)
                    }
                }

                impl planus::Primitive for BodyCompressionMethod {
                    const ALIGNMENT: usize = 1;
                    const SIZE: usize = 1;
                }

                impl planus::WriteAsPrimitive<BodyCompressionMethod> for BodyCompressionMethod {
                    #[inline]
                    fn write<const N: usize>(
                        &self,
                        cursor: planus::Cursor<'_, N>,
                        buffer_position: u32,
                    ) {
                        (*self as i8).write(cursor, buffer_position);
                    }
                }

                impl planus::WriteAs<BodyCompressionMethod> for BodyCompressionMethod {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(&self, _builder: &mut planus::Builder) -> BodyCompressionMethod {
                        *self
                    }
                }

                impl planus::WriteAsDefault<BodyCompressionMethod, BodyCompressionMethod>
                    for BodyCompressionMethod
                {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(
                        &self,
                        _builder: &mut planus::Builder,
                        default: &BodyCompressionMethod,
                    ) -> Option<BodyCompressionMethod> {
                        if self == default {
                            None
                        } else {
                            Some(*self)
                        }
                    }
                }

                impl planus::WriteAsOptional<BodyCompressionMethod> for BodyCompressionMethod {
                    type Prepared = Self;

                    #[inline]
                    fn prepare(
                        &self,
                        _builder: &mut planus::Builder,
                    ) -> Option<BodyCompressionMethod> {
                        Some(*self)
                    }
                }

                impl<'buf> planus::TableRead<'buf> for BodyCompressionMethod {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'buf>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        let n: i8 = planus::TableRead::from_buffer(buffer, offset)?;
                        Ok(n.try_into()?)
                    }
                }

                impl<'buf> planus::VectorRead<'buf> for BodyCompressionMethod {
                    type Output = std::result::Result<Self, planus::errors::UnknownEnumTag>;

                    const STRIDE: usize = 1;
                    #[inline]
                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'buf>,
                        offset: usize,
                    ) -> Self::Output {
                        let value = <i8 as planus::VectorRead>::from_buffer(buffer, offset);
                        let value: std::result::Result<Self, _> = value.try_into();
                        value.map_err(|error_kind| {
                            error_kind.with_error_location(
                                "BodyCompressionMethod",
                                "VectorRead::from_buffer",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl<'buf> planus::VectorWrite<BodyCompressionMethod> for BodyCompressionMethod {
                    const STRIDE: usize = 1;

                    type Value = Self;

                    fn prepare(&self, _builder: &mut planus::Builder) -> Self::Value {
                        *self
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[Self],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 1];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - i as u32,
                            );
                        }
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct BodyCompression {
                    pub codec: self::CompressionType,
                    pub method: self::BodyCompressionMethod,
                }

                impl BodyCompression {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(
                        builder: &mut planus::Builder,
                        codec: impl planus::WriteAsDefault<self::CompressionType, self::CompressionType>,
                        method: impl planus::WriteAsDefault<
                            self::BodyCompressionMethod,
                            self::BodyCompressionMethod,
                        >,
                    ) -> planus::Offset<Self> {
                        let prepared_codec =
                            codec.prepare(builder, &self::CompressionType::Lz4Frame);

                        let prepared_method =
                            method.prepare(builder, &self::BodyCompressionMethod::Buffer);

                        let mut table_writer =
                            planus::table_writer::TableWriter::<6, 2>::new(builder);

                        if prepared_codec.is_some() {
                            table_writer.calculate_size::<self::CompressionType>(2);
                        }
                        if prepared_method.is_some() {
                            table_writer.calculate_size::<self::BodyCompressionMethod>(4);
                        }

                        table_writer.finish_calculating();

                        unsafe {
                            if let Some(prepared_codec) = prepared_codec {
                                table_writer.write::<_, _, 1>(0, &prepared_codec);
                            }
                            if let Some(prepared_method) = prepared_method {
                                table_writer.write::<_, _, 1>(1, &prepared_method);
                            }
                        }

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<BodyCompression>> for BodyCompression {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> planus::Offset<BodyCompression> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<BodyCompression>> for BodyCompression {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<BodyCompression>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<BodyCompression> for BodyCompression {
                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> planus::Offset<BodyCompression> {
                        BodyCompression::create(builder, &self.codec, &self.method)
                    }
                }

                #[derive(Copy, Clone)]
                pub struct BodyCompressionRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> BodyCompressionRef<'a> {
                    pub fn codec(&self) -> planus::Result<self::CompressionType> {
                        Ok(self
                            .0
                            .access(0, "BodyCompression", "codec")?
                            .unwrap_or(self::CompressionType::Lz4Frame))
                    }

                    pub fn method(&self) -> planus::Result<self::BodyCompressionMethod> {
                        Ok(self
                            .0
                            .access(1, "BodyCompression", "method")?
                            .unwrap_or(self::BodyCompressionMethod::Buffer))
                    }
                }

                impl<'a> std::fmt::Debug for BodyCompressionRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("BodyCompressionRef");
                        if let Ok(codec) = self.codec() {
                            f.field("codec", &codec);
                        }
                        if let Ok(method) = self.method() {
                            f.field("method", &method);
                        }
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for BodyCompressionRef<'a> {
                    type Value = BodyCompression;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(BodyCompression {
                            codec: planus::ToOwned::to_owned(self.codec()?)?,
                            method: planus::ToOwned::to_owned(self.method()?)?,
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for BodyCompressionRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for BodyCompression {
                    type Output = planus::Result<BodyCompressionRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[BodyCompressionRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<BodyCompression>> for BodyCompression {
                    type Value = planus::Offset<BodyCompression>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<BodyCompression>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for BodyCompressionRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[BodyCompressionRef]",
                                "read_as_root",
                                0,
                            )
                        })
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct RecordBatch {
                    pub length: i64,
                    pub nodes: Option<Vec<self::FieldNode>>,
                    pub buffers: Option<Vec<self::Buffer>>,
                    pub compression: Option<Box<self::BodyCompression>>,
                }

                impl RecordBatch {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(
                        builder: &mut planus::Builder,
                        length: impl planus::WriteAsDefault<i64, i64>,
                        nodes: impl planus::WriteAsOptional<planus::Offset<[self::FieldNode]>>,
                        buffers: impl planus::WriteAsOptional<planus::Offset<[self::Buffer]>>,
                        compression: impl planus::WriteAsOptional<planus::Offset<self::BodyCompression>>,
                    ) -> planus::Offset<Self> {
                        let prepared_length = length.prepare(builder, &0);

                        let prepared_nodes = nodes.prepare(builder);

                        let prepared_buffers = buffers.prepare(builder);

                        let prepared_compression = compression.prepare(builder);

                        let mut table_writer =
                            planus::table_writer::TableWriter::<10, 20>::new(builder);

                        if prepared_length.is_some() {
                            table_writer.calculate_size::<i64>(2);
                        }
                        if prepared_nodes.is_some() {
                            table_writer.calculate_size::<planus::Offset<[self::FieldNode]>>(4);
                        }
                        if prepared_buffers.is_some() {
                            table_writer.calculate_size::<planus::Offset<[self::Buffer]>>(6);
                        }
                        if prepared_compression.is_some() {
                            table_writer.calculate_size::<planus::Offset<self::BodyCompression>>(8);
                        }

                        table_writer.finish_calculating();

                        unsafe {
                            if let Some(prepared_length) = prepared_length {
                                table_writer.write::<_, _, 8>(0, &prepared_length);
                            }
                            if let Some(prepared_nodes) = prepared_nodes {
                                table_writer.write::<_, _, 4>(1, &prepared_nodes);
                            }
                            if let Some(prepared_buffers) = prepared_buffers {
                                table_writer.write::<_, _, 4>(2, &prepared_buffers);
                            }
                            if let Some(prepared_compression) = prepared_compression {
                                table_writer.write::<_, _, 4>(3, &prepared_compression);
                            }
                        }

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<RecordBatch>> for RecordBatch {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> planus::Offset<RecordBatch> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<RecordBatch>> for RecordBatch {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<RecordBatch>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<RecordBatch> for RecordBatch {
                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> planus::Offset<RecordBatch> {
                        RecordBatch::create(
                            builder,
                            &self.length,
                            &self.nodes,
                            &self.buffers,
                            &self.compression,
                        )
                    }
                }

                #[derive(Copy, Clone)]
                pub struct RecordBatchRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> RecordBatchRef<'a> {
                    pub fn length(&self) -> planus::Result<i64> {
                        Ok(self.0.access(0, "RecordBatch", "length")?.unwrap_or(0))
                    }

                    pub fn nodes(
                        &self,
                    ) -> planus::Result<Option<planus::Vector<'a, self::FieldNode>>>
                    {
                        self.0.access(1, "RecordBatch", "nodes")
                    }

                    pub fn buffers(
                        &self,
                    ) -> planus::Result<Option<planus::Vector<'a, self::Buffer>>>
                    {
                        self.0.access(2, "RecordBatch", "buffers")
                    }

                    pub fn compression(
                        &self,
                    ) -> planus::Result<Option<self::BodyCompressionRef<'a>>> {
                        self.0.access(3, "RecordBatch", "compression")
                    }
                }

                impl<'a> std::fmt::Debug for RecordBatchRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("RecordBatchRef");
                        if let Ok(length) = self.length() {
                            f.field("length", &length);
                        }
                        if let Ok(Some(nodes)) = self.nodes() {
                            f.field("nodes", &nodes);
                        }
                        if let Ok(Some(buffers)) = self.buffers() {
                            f.field("buffers", &buffers);
                        }
                        if let Ok(Some(compression)) = self.compression() {
                            f.field("compression", &compression);
                        }
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for RecordBatchRef<'a> {
                    type Value = RecordBatch;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(RecordBatch {
                            length: planus::ToOwned::to_owned(self.length()?)?,
                            nodes: if let Some(nodes) = self.nodes()? {
                                Some(planus::ToOwned::to_owned(nodes)?)
                            } else {
                                None
                            },
                            buffers: if let Some(buffers) = self.buffers()? {
                                Some(planus::ToOwned::to_owned(buffers)?)
                            } else {
                                None
                            },
                            compression: if let Some(compression) = self.compression()? {
                                Some(Box::new(planus::ToOwned::to_owned(compression)?))
                            } else {
                                None
                            },
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for RecordBatchRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for RecordBatch {
                    type Output = planus::Result<RecordBatchRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[RecordBatchRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<RecordBatch>> for RecordBatch {
                    type Value = planus::Offset<RecordBatch>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<RecordBatch>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for RecordBatchRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location("[RecordBatchRef]", "read_as_root", 0)
                        })
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct DictionaryBatch {
                    pub id: i64,
                    pub data: Option<Box<self::RecordBatch>>,
                    pub is_delta: bool,
                }

                impl DictionaryBatch {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(
                        builder: &mut planus::Builder,
                        id: impl planus::WriteAsDefault<i64, i64>,
                        data: impl planus::WriteAsOptional<planus::Offset<self::RecordBatch>>,
                        is_delta: impl planus::WriteAsDefault<bool, bool>,
                    ) -> planus::Offset<Self> {
                        let prepared_id = id.prepare(builder, &0);

                        let prepared_data = data.prepare(builder);

                        let prepared_is_delta = is_delta.prepare(builder, &false);

                        let mut table_writer =
                            planus::table_writer::TableWriter::<8, 13>::new(builder);

                        if prepared_id.is_some() {
                            table_writer.calculate_size::<i64>(2);
                        }
                        if prepared_data.is_some() {
                            table_writer.calculate_size::<planus::Offset<self::RecordBatch>>(4);
                        }
                        if prepared_is_delta.is_some() {
                            table_writer.calculate_size::<bool>(6);
                        }

                        table_writer.finish_calculating();

                        unsafe {
                            if let Some(prepared_id) = prepared_id {
                                table_writer.write::<_, _, 8>(0, &prepared_id);
                            }
                            if let Some(prepared_data) = prepared_data {
                                table_writer.write::<_, _, 4>(1, &prepared_data);
                            }
                            if let Some(prepared_is_delta) = prepared_is_delta {
                                table_writer.write::<_, _, 1>(2, &prepared_is_delta);
                            }
                        }

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<DictionaryBatch>> for DictionaryBatch {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> planus::Offset<DictionaryBatch> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<DictionaryBatch>> for DictionaryBatch {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<DictionaryBatch>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<DictionaryBatch> for DictionaryBatch {
                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> planus::Offset<DictionaryBatch> {
                        DictionaryBatch::create(builder, &self.id, &self.data, &self.is_delta)
                    }
                }

                #[derive(Copy, Clone)]
                pub struct DictionaryBatchRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> DictionaryBatchRef<'a> {
                    pub fn id(&self) -> planus::Result<i64> {
                        Ok(self.0.access(0, "DictionaryBatch", "id")?.unwrap_or(0))
                    }

                    pub fn data(&self) -> planus::Result<Option<self::RecordBatchRef<'a>>> {
                        self.0.access(1, "DictionaryBatch", "data")
                    }

                    pub fn is_delta(&self) -> planus::Result<bool> {
                        Ok(self
                            .0
                            .access(2, "DictionaryBatch", "is_delta")?
                            .unwrap_or(false))
                    }
                }

                impl<'a> std::fmt::Debug for DictionaryBatchRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("DictionaryBatchRef");
                        if let Ok(id) = self.id() {
                            f.field("id", &id);
                        }
                        if let Ok(Some(data)) = self.data() {
                            f.field("data", &data);
                        }
                        if let Ok(is_delta) = self.is_delta() {
                            f.field("is_delta", &is_delta);
                        }
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for DictionaryBatchRef<'a> {
                    type Value = DictionaryBatch;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(DictionaryBatch {
                            id: planus::ToOwned::to_owned(self.id()?)?,
                            data: if let Some(data) = self.data()? {
                                Some(Box::new(planus::ToOwned::to_owned(data)?))
                            } else {
                                None
                            },
                            is_delta: planus::ToOwned::to_owned(self.is_delta()?)?,
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for DictionaryBatchRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for DictionaryBatch {
                    type Output = planus::Result<DictionaryBatchRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[DictionaryBatchRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<DictionaryBatch>> for DictionaryBatch {
                    type Value = planus::Offset<DictionaryBatch>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<DictionaryBatch>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for DictionaryBatchRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[DictionaryBatchRef]",
                                "read_as_root",
                                0,
                            )
                        })
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub enum MessageHeader {
                    Schema(Box<self::Schema>),
                    DictionaryBatch(Box<self::DictionaryBatch>),
                    RecordBatch(Box<self::RecordBatch>),
                    Tensor(Box<self::Tensor>),
                    SparseTensor(Box<self::SparseTensor>),
                }

                impl MessageHeader {
                    pub fn create_schema(
                        builder: &mut planus::Builder,
                        value: impl planus::WriteAsOffset<self::Schema>,
                    ) -> planus::UnionOffset<Self> {
                        planus::UnionOffset::new(1, value.prepare(builder).downcast())
                    }

                    pub fn create_dictionary_batch(
                        builder: &mut planus::Builder,
                        value: impl planus::WriteAsOffset<self::DictionaryBatch>,
                    ) -> planus::UnionOffset<Self> {
                        planus::UnionOffset::new(2, value.prepare(builder).downcast())
                    }

                    pub fn create_record_batch(
                        builder: &mut planus::Builder,
                        value: impl planus::WriteAsOffset<self::RecordBatch>,
                    ) -> planus::UnionOffset<Self> {
                        planus::UnionOffset::new(3, value.prepare(builder).downcast())
                    }

                    pub fn create_tensor(
                        builder: &mut planus::Builder,
                        value: impl planus::WriteAsOffset<self::Tensor>,
                    ) -> planus::UnionOffset<Self> {
                        planus::UnionOffset::new(4, value.prepare(builder).downcast())
                    }

                    pub fn create_sparse_tensor(
                        builder: &mut planus::Builder,
                        value: impl planus::WriteAsOffset<self::SparseTensor>,
                    ) -> planus::UnionOffset<Self> {
                        planus::UnionOffset::new(5, value.prepare(builder).downcast())
                    }
                }

                impl planus::WriteAsUnion<MessageHeader> for MessageHeader {
                    fn prepare(&self, builder: &mut planus::Builder) -> planus::UnionOffset<Self> {
                        match self {
                            Self::Schema(value) => Self::create_schema(builder, value),
                            Self::DictionaryBatch(value) => {
                                Self::create_dictionary_batch(builder, value)
                            }
                            Self::RecordBatch(value) => Self::create_record_batch(builder, value),
                            Self::Tensor(value) => Self::create_tensor(builder, value),
                            Self::SparseTensor(value) => Self::create_sparse_tensor(builder, value),
                        }
                    }
                }

                impl planus::WriteAsOptionalUnion<MessageHeader> for MessageHeader {
                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::UnionOffset<Self>> {
                        Some(planus::WriteAsUnion::prepare(self, builder))
                    }
                }

                #[derive(Copy, Clone, Debug)]
                pub enum MessageHeaderRef<'a> {
                    Schema(self::SchemaRef<'a>),
                    DictionaryBatch(self::DictionaryBatchRef<'a>),
                    RecordBatch(self::RecordBatchRef<'a>),
                    Tensor(self::TensorRef<'a>),
                    SparseTensor(self::SparseTensorRef<'a>),
                }

                impl<'a> planus::ToOwned for MessageHeaderRef<'a> {
                    type Value = MessageHeader;

                    fn to_owned(self) -> planus::Result<MessageHeader> {
                        Ok(match self {
                            Self::Schema(value) => {
                                MessageHeader::Schema(Box::new(planus::ToOwned::to_owned(value)?))
                            }

                            Self::DictionaryBatch(value) => MessageHeader::DictionaryBatch(
                                Box::new(planus::ToOwned::to_owned(value)?),
                            ),

                            Self::RecordBatch(value) => MessageHeader::RecordBatch(Box::new(
                                planus::ToOwned::to_owned(value)?,
                            )),

                            Self::Tensor(value) => {
                                MessageHeader::Tensor(Box::new(planus::ToOwned::to_owned(value)?))
                            }

                            Self::SparseTensor(value) => MessageHeader::SparseTensor(Box::new(
                                planus::ToOwned::to_owned(value)?,
                            )),
                        })
                    }
                }

                impl<'a> planus::TableReadUnion<'a> for MessageHeaderRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        field_offset: usize,
                        tag: u8,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        match tag {
                            1 => Ok(Self::Schema(planus::TableRead::from_buffer(
                                buffer,
                                field_offset,
                            )?)),
                            2 => Ok(Self::DictionaryBatch(planus::TableRead::from_buffer(
                                buffer,
                                field_offset,
                            )?)),
                            3 => Ok(Self::RecordBatch(planus::TableRead::from_buffer(
                                buffer,
                                field_offset,
                            )?)),
                            4 => Ok(Self::Tensor(planus::TableRead::from_buffer(
                                buffer,
                                field_offset,
                            )?)),
                            5 => Ok(Self::SparseTensor(planus::TableRead::from_buffer(
                                buffer,
                                field_offset,
                            )?)),
                            _ => Err(planus::errors::ErrorKind::UnknownUnionTag { tag }),
                        }
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct Message {
                    pub version: self::MetadataVersion,
                    pub header: Option<self::MessageHeader>,
                    pub body_length: i64,
                    pub custom_metadata: Option<Vec<self::KeyValue>>,
                }

                impl Message {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(
                        builder: &mut planus::Builder,
                        version: impl planus::WriteAsDefault<
                            self::MetadataVersion,
                            self::MetadataVersion,
                        >,
                        header: impl planus::WriteAsOptionalUnion<self::MessageHeader>,
                        body_length: impl planus::WriteAsDefault<i64, i64>,
                        custom_metadata: impl planus::WriteAsOptional<
                            planus::Offset<[planus::Offset<self::KeyValue>]>,
                        >,
                    ) -> planus::Offset<Self> {
                        let prepared_version = version.prepare(builder, &self::MetadataVersion::V1);

                        let prepared_header = header.prepare(builder);

                        let prepared_body_length = body_length.prepare(builder, &0);

                        let prepared_custom_metadata = custom_metadata.prepare(builder);

                        let mut table_writer =
                            planus::table_writer::TableWriter::<12, 19>::new(builder);

                        if prepared_version.is_some() {
                            table_writer.calculate_size::<self::MetadataVersion>(2);
                        }
                        if prepared_header.is_some() {
                            table_writer.calculate_size::<u8>(4);
                            table_writer.calculate_size::<planus::Offset<self::MessageHeader>>(6);
                        }
                        if prepared_body_length.is_some() {
                            table_writer.calculate_size::<i64>(8);
                        }
                        if prepared_custom_metadata.is_some() {
                            table_writer
                                .calculate_size::<planus::Offset<[planus::Offset<self::KeyValue>]>>(
                                    10,
                                );
                        }

                        table_writer.finish_calculating();

                        unsafe {
                            if let Some(prepared_body_length) = prepared_body_length {
                                table_writer.write::<_, _, 8>(3, &prepared_body_length);
                            }
                            if let Some(prepared_header) = prepared_header {
                                table_writer.write::<_, _, 4>(2, &prepared_header.offset);
                            }
                            if let Some(prepared_custom_metadata) = prepared_custom_metadata {
                                table_writer.write::<_, _, 4>(4, &prepared_custom_metadata);
                            }
                            if let Some(prepared_version) = prepared_version {
                                table_writer.write::<_, _, 2>(0, &prepared_version);
                            }
                            if let Some(prepared_header) = prepared_header {
                                table_writer.write::<_, _, 1>(1, &prepared_header.tag);
                            }
                        }

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<Message>> for Message {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Message> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<Message>> for Message {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<Message>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<Message> for Message {
                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Message> {
                        Message::create(
                            builder,
                            &self.version,
                            &self.header,
                            &self.body_length,
                            &self.custom_metadata,
                        )
                    }
                }

                #[derive(Copy, Clone)]
                pub struct MessageRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> MessageRef<'a> {
                    pub fn version(&self) -> planus::Result<self::MetadataVersion> {
                        Ok(self
                            .0
                            .access(0, "Message", "version")?
                            .unwrap_or(self::MetadataVersion::V1))
                    }

                    pub fn header(&self) -> planus::Result<Option<self::MessageHeaderRef<'a>>> {
                        self.0.access_union(1, "Message", "header")
                    }

                    pub fn body_length(&self) -> planus::Result<i64> {
                        Ok(self.0.access(3, "Message", "body_length")?.unwrap_or(0))
                    }

                    pub fn custom_metadata(
                        &self,
                    ) -> planus::Result<Option<planus::Vector<'a, self::KeyValue>>>
                    {
                        self.0.access(4, "Message", "custom_metadata")
                    }
                }

                impl<'a> std::fmt::Debug for MessageRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("MessageRef");
                        if let Ok(version) = self.version() {
                            f.field("version", &version);
                        }
                        if let Ok(Some(header)) = self.header() {
                            f.field("header", &header);
                        }
                        if let Ok(body_length) = self.body_length() {
                            f.field("body_length", &body_length);
                        }
                        if let Ok(Some(custom_metadata)) = self.custom_metadata() {
                            f.field("custom_metadata", &custom_metadata);
                        }
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for MessageRef<'a> {
                    type Value = Message;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(Message {
                            version: planus::ToOwned::to_owned(self.version()?)?,
                            header: if let Some(header) = self.header()? {
                                Some(planus::ToOwned::to_owned(header)?)
                            } else {
                                None
                            },
                            body_length: planus::ToOwned::to_owned(self.body_length()?)?,
                            custom_metadata: if let Some(custom_metadata) =
                                self.custom_metadata()?
                            {
                                Some(planus::ToOwned::to_owned(custom_metadata)?)
                            } else {
                                None
                            },
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for MessageRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for Message {
                    type Output = planus::Result<MessageRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[MessageRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<Message>> for Message {
                    type Value = planus::Offset<Message>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<Message>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for MessageRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location("[MessageRef]", "read_as_root", 0)
                        })
                    }
                }

                #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
                pub struct Footer {
                    pub version: self::MetadataVersion,
                    pub schema: Option<Box<self::Schema>>,
                    pub dictionaries: Option<Vec<self::Block>>,
                    pub record_batches: Option<Vec<self::Block>>,
                    pub custom_metadata: Option<Vec<self::KeyValue>>,
                }

                impl Footer {
                    #[allow(clippy::too_many_arguments)]
                    pub fn create(
                        builder: &mut planus::Builder,
                        version: impl planus::WriteAsDefault<
                            self::MetadataVersion,
                            self::MetadataVersion,
                        >,
                        schema: impl planus::WriteAsOptional<planus::Offset<self::Schema>>,
                        dictionaries: impl planus::WriteAsOptional<planus::Offset<[self::Block]>>,
                        record_batches: impl planus::WriteAsOptional<planus::Offset<[self::Block]>>,
                        custom_metadata: impl planus::WriteAsOptional<
                            planus::Offset<[planus::Offset<self::KeyValue>]>,
                        >,
                    ) -> planus::Offset<Self> {
                        let prepared_version = version.prepare(builder, &self::MetadataVersion::V1);

                        let prepared_schema = schema.prepare(builder);

                        let prepared_dictionaries = dictionaries.prepare(builder);

                        let prepared_record_batches = record_batches.prepare(builder);

                        let prepared_custom_metadata = custom_metadata.prepare(builder);

                        let mut table_writer =
                            planus::table_writer::TableWriter::<12, 18>::new(builder);

                        if prepared_version.is_some() {
                            table_writer.calculate_size::<self::MetadataVersion>(2);
                        }
                        if prepared_schema.is_some() {
                            table_writer.calculate_size::<planus::Offset<self::Schema>>(4);
                        }
                        if prepared_dictionaries.is_some() {
                            table_writer.calculate_size::<planus::Offset<[self::Block]>>(6);
                        }
                        if prepared_record_batches.is_some() {
                            table_writer.calculate_size::<planus::Offset<[self::Block]>>(8);
                        }
                        if prepared_custom_metadata.is_some() {
                            table_writer
                                .calculate_size::<planus::Offset<[planus::Offset<self::KeyValue>]>>(
                                    10,
                                );
                        }

                        table_writer.finish_calculating();

                        unsafe {
                            if let Some(prepared_schema) = prepared_schema {
                                table_writer.write::<_, _, 4>(1, &prepared_schema);
                            }
                            if let Some(prepared_dictionaries) = prepared_dictionaries {
                                table_writer.write::<_, _, 4>(2, &prepared_dictionaries);
                            }
                            if let Some(prepared_record_batches) = prepared_record_batches {
                                table_writer.write::<_, _, 4>(3, &prepared_record_batches);
                            }
                            if let Some(prepared_custom_metadata) = prepared_custom_metadata {
                                table_writer.write::<_, _, 4>(4, &prepared_custom_metadata);
                            }
                            if let Some(prepared_version) = prepared_version {
                                table_writer.write::<_, _, 2>(0, &prepared_version);
                            }
                        }

                        table_writer.finish()
                    }
                }

                impl planus::WriteAs<planus::Offset<Footer>> for Footer {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Footer> {
                        planus::WriteAsOffset::prepare(self, builder)
                    }
                }

                impl planus::WriteAsOptional<planus::Offset<Footer>> for Footer {
                    type Prepared = planus::Offset<Self>;

                    fn prepare(
                        &self,
                        builder: &mut planus::Builder,
                    ) -> Option<planus::Offset<Footer>> {
                        Some(planus::WriteAsOffset::prepare(self, builder))
                    }
                }

                impl planus::WriteAsOffset<Footer> for Footer {
                    fn prepare(&self, builder: &mut planus::Builder) -> planus::Offset<Footer> {
                        Footer::create(
                            builder,
                            &self.version,
                            &self.schema,
                            &self.dictionaries,
                            &self.record_batches,
                            &self.custom_metadata,
                        )
                    }
                }

                #[derive(Copy, Clone)]
                pub struct FooterRef<'a>(planus::table_reader::Table<'a>);

                impl<'a> FooterRef<'a> {
                    pub fn version(&self) -> planus::Result<self::MetadataVersion> {
                        Ok(self
                            .0
                            .access(0, "Footer", "version")?
                            .unwrap_or(self::MetadataVersion::V1))
                    }

                    pub fn schema(&self) -> planus::Result<Option<self::SchemaRef<'a>>> {
                        self.0.access(1, "Footer", "schema")
                    }

                    pub fn dictionaries(
                        &self,
                    ) -> planus::Result<Option<planus::Vector<'a, self::Block>>>
                    {
                        self.0.access(2, "Footer", "dictionaries")
                    }

                    pub fn record_batches(
                        &self,
                    ) -> planus::Result<Option<planus::Vector<'a, self::Block>>>
                    {
                        self.0.access(3, "Footer", "record_batches")
                    }

                    pub fn custom_metadata(
                        &self,
                    ) -> planus::Result<Option<planus::Vector<'a, self::KeyValue>>>
                    {
                        self.0.access(4, "Footer", "custom_metadata")
                    }
                }

                impl<'a> std::fmt::Debug for FooterRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("FooterRef");
                        if let Ok(version) = self.version() {
                            f.field("version", &version);
                        }
                        if let Ok(Some(schema)) = self.schema() {
                            f.field("schema", &schema);
                        }
                        if let Ok(Some(dictionaries)) = self.dictionaries() {
                            f.field("dictionaries", &dictionaries);
                        }
                        if let Ok(Some(record_batches)) = self.record_batches() {
                            f.field("record_batches", &record_batches);
                        }
                        if let Ok(Some(custom_metadata)) = self.custom_metadata() {
                            f.field("custom_metadata", &custom_metadata);
                        }
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for FooterRef<'a> {
                    type Value = Footer;

                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(Footer {
                            version: planus::ToOwned::to_owned(self.version()?)?,
                            schema: if let Some(schema) = self.schema()? {
                                Some(Box::new(planus::ToOwned::to_owned(schema)?))
                            } else {
                                None
                            },
                            dictionaries: if let Some(dictionaries) = self.dictionaries()? {
                                Some(planus::ToOwned::to_owned(dictionaries)?)
                            } else {
                                None
                            },
                            record_batches: if let Some(record_batches) = self.record_batches()? {
                                Some(planus::ToOwned::to_owned(record_batches)?)
                            } else {
                                None
                            },
                            custom_metadata: if let Some(custom_metadata) =
                                self.custom_metadata()?
                            {
                                Some(planus::ToOwned::to_owned(custom_metadata)?)
                            } else {
                                None
                            },
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for FooterRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        Ok(Self(planus::table_reader::Table::from_buffer(
                            buffer, offset,
                        )?))
                    }
                }

                impl<'a> planus::VectorRead<'a> for Footer {
                    type Output = planus::Result<FooterRef<'a>>;
                    const STRIDE: usize = 4;

                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> Self::Output {
                        planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                            error_kind.with_error_location(
                                "[FooterRef]",
                                "get",
                                buffer.offset_from_start,
                            )
                        })
                    }
                }

                impl planus::VectorWrite<planus::Offset<Footer>> for Footer {
                    type Value = planus::Offset<Footer>;
                    const STRIDE: usize = 4;
                    fn prepare(&self, builder: &mut planus::Builder) -> Self::Value {
                        planus::WriteAs::prepare(self, builder)
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[planus::Offset<Footer>],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 4];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (Self::STRIDE * i) as u32,
                            );
                        }
                    }
                }

                impl<'a> planus::ReadAsRoot<'a> for FooterRef<'a> {
                    fn read_as_root(slice: &'a [u8]) -> planus::Result<Self> {
                        planus::TableRead::from_buffer(
                            planus::SliceWithStartOffset {
                                buffer: slice,
                                offset_from_start: 0,
                            },
                            0,
                        )
                        .map_err(|error_kind| {
                            error_kind.with_error_location("[FooterRef]", "read_as_root", 0)
                        })
                    }
                }

                #[derive(Copy, Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
                pub struct Block {
                    pub offset: i64,
                    pub meta_data_length: i32,
                    pub body_length: i64,
                }

                impl planus::Primitive for Block {
                    const ALIGNMENT: usize = 8;
                    const SIZE: usize = 24;
                }

                #[allow(clippy::identity_op)]
                impl planus::WriteAsPrimitive<Block> for Block {
                    fn write<const N: usize>(
                        &self,
                        cursor: planus::Cursor<'_, N>,
                        buffer_position: u32,
                    ) {
                        let (cur, cursor) = cursor.split::<8, 16>();
                        self.offset.write(cur, buffer_position - 0);
                        let (cur, cursor) = cursor.split::<4, 12>();
                        self.meta_data_length.write(cur, buffer_position - 8);
                        let cursor = cursor.write::<4, 8>([0; 4]);
                        let (cur, cursor) = cursor.split::<8, 0>();
                        self.body_length.write(cur, buffer_position - 16);
                        cursor.finish([]);
                    }
                }

                impl planus::WriteAs<Block> for Block {
                    type Prepared = Self;
                    fn prepare(&self, _builder: &mut planus::Builder) -> Self {
                        *self
                    }
                }

                impl planus::WriteAsOptional<Block> for Block {
                    type Prepared = Self;
                    fn prepare(&self, _builder: &mut planus::Builder) -> Option<Self> {
                        Some(*self)
                    }
                }

                #[derive(Copy, Clone)]
                pub struct BlockRef<'a>(planus::ArrayWithStartOffset<'a, 24>);

                impl<'a> BlockRef<'a> {
                    pub fn offset(&self) -> i64 {
                        let buffer = self.0.advance_as_array::<8>(0).unwrap();

                        i64::from_le_bytes(*buffer.as_array())
                    }

                    pub fn meta_data_length(&self) -> i32 {
                        let buffer = self.0.advance_as_array::<4>(8).unwrap();

                        i32::from_le_bytes(*buffer.as_array())
                    }

                    pub fn body_length(&self) -> i64 {
                        let buffer = self.0.advance_as_array::<8>(16).unwrap();

                        i64::from_le_bytes(*buffer.as_array())
                    }
                }

                impl<'a> std::fmt::Debug for BlockRef<'a> {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let mut f = f.debug_struct("BlockRef");
                        f.field("offset", &self.offset());
                        f.field("meta_data_length", &self.meta_data_length());
                        f.field("body_length", &self.body_length());
                        f.finish()
                    }
                }

                impl<'a> planus::ToOwned for BlockRef<'a> {
                    type Value = Block;
                    fn to_owned(self) -> planus::Result<Self::Value> {
                        Ok(Block {
                            offset: self.offset(),
                            meta_data_length: self.meta_data_length(),
                            body_length: self.body_length(),
                        })
                    }
                }

                impl<'a> planus::TableRead<'a> for BlockRef<'a> {
                    fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> std::result::Result<Self, planus::errors::ErrorKind> {
                        let buffer = buffer.advance_as_array::<24>(offset)?;
                        Ok(Self(buffer))
                    }
                }

                impl<'a> planus::VectorRead<'a> for Block {
                    const STRIDE: usize = 24;

                    type Output = BlockRef<'a>;
                    unsafe fn from_buffer(
                        buffer: planus::SliceWithStartOffset<'a>,
                        offset: usize,
                    ) -> BlockRef<'a> {
                        BlockRef(buffer.unchecked_advance_as_array(offset))
                    }
                }

                impl planus::VectorWrite<Block> for Block {
                    const STRIDE: usize = 24;

                    type Value = Block;

                    fn prepare(&self, _builder: &mut planus::Builder) -> Self::Value {
                        *self
                    }

                    #[inline]
                    unsafe fn write_values(
                        values: &[Block],
                        bytes: *mut std::mem::MaybeUninit<u8>,
                        buffer_position: u32,
                    ) {
                        let bytes = bytes as *mut [std::mem::MaybeUninit<u8>; 24];
                        for (i, v) in values.iter().enumerate() {
                            planus::WriteAsPrimitive::write(
                                v,
                                planus::Cursor::new(&mut *bytes.add(i)),
                                buffer_position - (24 * i) as u32,
                            );
                        }
                    }
                }
            }
        }
    }
}
