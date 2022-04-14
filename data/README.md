## Here is an example of pre-training data. There are two files "raw_multidata_of_product_preatrain.small_train" and "raw_multidata_of_product_preatrain.small_valid", containing about 500 samples. Each line contains 5 fields: Item ID, Item title, Item image url, Item KG, Item category (which is not used when pre-training K3M model).

## Take one line of data as an example: "631432564317	家用户型圆桌圆形折叠桌简约餐桌老式简易桌子4人8人小吃饭桌子大	https://img.alicdn.com/imgextra/https://img.alicdn.com/imgextra/i1/2209527442205/O1CN014C9neM1S9vxl3cyha_!!0-item_pic.jpg	省份#:#河北省#;#款式定位#:#经济型#;#附加功能#:#多功能#;#人造板种类#:#密度板/纤维板#;#地市#:#廊坊市#;#区县#:#安次区#;#材质#:#人造板#;#是否可定制#:#否#;#出租车是否可运输#:#是#;#风格#:#简约现代	折叠桌"
## where,
- Item ID: 631432564317
- Item title: 家用户型圆桌圆形折叠桌简约餐桌老式简易桌子4人8人小吃饭桌子大	
- Item image url: https://img.alicdn.com/imgextra/https://img.alicdn.com/imgextra/i1/2209527442205/O1CN014C9neM1S9vxl3cyha_!!0-item_pic.jpg
- Item KG: 省份#:#河北省#;#款式定位#:#经济型#;#附加功能#:#多功能#;#人造板种类#:#密度板/纤维板#;#地市#:#廊坊市#;#区县#:#安次区#;#材质#:#人造板#;#是否可定制#:#否#;#出租车是否可运输#:#是#;#风格#:#简约现代
- Item category: 折叠桌

## Note: If you need your own data to pretrain the model, please construct the data in the above format. The five fields in each row are separated by '\t'. 
## The "Item KG" field consists of "property-value" pairs of item, and different "property-value" pairs are separated by '#;#', the property and its corresponding value are separated by '#:#'. In the example, the property-value pair "省份#:#河北省" represents a triple <item, 省份, 河北省>. 
## Since the "Item category" field is not used in the pre-training stage, it can be set to any value in your own data.