for NAME in Tensor SparseTensor Schema Message File
do
	curl https://raw.githubusercontent.com/apache/arrow/master/format/$NAME.fbs -o $NAME.fbs
done

planus rust Tensor.fbs SparseTensor.fbs Schema.fbs Message.fbs File.fbs -o src/ipc/generated.rs
