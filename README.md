# 風格轉移 & DeepDream 學習筆記

## 目錄
- [神經風格轉移](#神經風格轉移)
  - [核心概念](#核心概念)
  - [Loss 函數詳解](#loss-函數詳解)
  - [實現流程](#實現流程)
- [DeepDream](#deepdream)

---

## 風格轉移

### 核心概念

**目標：** 將一張圖片的「風格」應用到另一張圖片的「內容」上

**例子：**
- 內容圖像 = 你的照片（人物、建築等）
- 風格圖像 = 梵谷的《星月夜》（筆觸、色彩、質感）
- 結果 = 你的照片 + 梵谷風格

**核心原理：**
使用預訓練的 VGG19 神經網絡在不同層級提取特徵：
- **深層（如 block5_conv2）** → 提取「高級內容」（物體、結構）
- **淺層（如 block1_conv1 ~ block5_conv1）** → 提取「低級風格」（紋理、色彩、筆觸）

---

### Loss 函數詳解

#### 1. 內容損失（Content Loss）

```python
def content_loss(base_img, combination_img):
    return tf.reduce_sum(tf.square(combination_img - base_img))
```

**作用：** 保留原始圖像的內容

**在 compute_loss 中的使用：**
```python
layer_features = features[content_layer_name]  # block5_conv2（深層）
base_image_features = layer_features[0, :, :, :]        # 內容圖像特徵
combination_features = layer_features[2, :, :, :]       # 組合圖像特徵

loss = loss + content_weight * content_loss(
    base_image_features, combination_features
)
```

**邏輯：**
- 在 **深層** 比較特徵
- 確保生成圖像保持原始圖像的物體、結構、輪廓
- `content_weight = 2.5e-8`（很小，因為我們主要要改風格）

---

#### 2. 風格損失（Style Loss）

```python
def gram_matrix(x):
    """計算 Gram 矩陣 - 捕捉圖像風格"""
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

def style_loss(style_img, combination_img):
    """比較 Gram 矩陣來度量風格差異"""
    S = gram_matrix(style_img)      # 風格圖像的 Gram 矩陣
    C = gram_matrix(combination_img)  # 組合圖像的 Gram 矩陣
    channels = 3
    size = img_height * img_width
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))
```

**為什麼用 Gram 矩陣？**
- Gram 矩陣捕捉特徵之間的相關性（correlation）
- 代表了「風格」：色彩分佈、紋理、筆觸方向等
- 不依賴空間位置，只關心風格特徵的組合

**在 compute_loss 中的使用：**
```python
for layer_name in style_layer_names:  # 多層淺層特徵
    layer_features = features[layer_name]
    style_reference_features = layer_features[1, :, :, :]    # 風格圖像特徵
    combination_features = layer_features[2, :, :, :]        # 組合圖像特徵

    style_loss_value = style_loss(
        style_reference_features, combination_features
    )
    loss += (style_weight / len(style_layer_names)) * style_loss_value
```

**邏輯：**
- 在 **多個淺層** 比較特徵（block1_conv1 到 block5_conv1）
- 多層保證捕捉不同粒度的風格（細節→粗糙）
- `style_weight = 1e-6`（很小，防止過度風格化）
- 除以層數做平均，防止某一層主導損失

---

#### 3. 總變差損失（Total Variation Loss）

```python
def total_variation_loss(x):
    """減少圖像噪聲，使相鄰像素更接近"""
    a = tf.square(
        x[:, : img_height - 1, : img_width - 1, :] -
        x[:, 1:, : img_width - 1, :]  # 垂直方向差異
    )
    b = tf.square(
        x[:, : img_height - 1, : img_width - 1, :] -
        x[:, : img_height - 1, 1:, :]  # 水平方向差異
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))
```

**作用：** 使生成圖像更平滑，減少噪聲和雜點

**邏輯：**
- 計算相鄰像素的差異
- 差異越大，損失越高
- 優化器會最小化這個損失，使圖像變光滑
- `total_variation_weight = 1e-6`（很小，只是輔助平滑）

---

#### 4. 完整的 compute_loss 函數

```python
def compute_loss(combination_image, base_image, style_reference_image):
    # 三張圖像一起輸入，提高計算效率
    # [0] = base_image (內容)
    # [1] = style_reference_image (風格)
    # [2] = combination_image (優化目標)
    input_tensor = tf.concat(
        [base_image, style_reference_image, combination_image], axis=0
    )

    # 一次通過神經網絡，提取所有層的特徵
    features = feature_extractor(input_tensor)
    loss = tf.zeros(shape=())

    # ===== 內容損失：保持內容 =====
    layer_features = features[content_layer_name]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(
        base_image_features, combination_features
    )

    # ===== 風格損失：採用風格 =====
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        style_loss_value = style_loss(
            style_reference_features, combination_features
        )
        loss += (style_weight / len(style_layer_names)) * style_loss_value

    # ===== 總變差損失：減少噪聲 =====
    loss += total_variation_weight * total_variation_loss(combination_image)

    return loss
```

---

### 實現流程

#### 訓練迴圈

```python
iterations = 4000
for i in range(1, iterations + 1):
    # 1. 計算損失和梯度
    loss, grads = compute_loss_and_grads(
        combination_image, base_image, style_reference_image
    )

    # 2. 更新組合圖像像素
    optimizer.apply_gradients([(grads, combination_image)])

    # 3. 每100次迭代保存進度
    if i % 100 == 0:
        print(f"Iteration {i}: loss={loss:.2f}")
        img = deprocess_image(combination_image.numpy())
        fname = f"combination_image_at_iteration_{i}.png"
        keras.utils.save_img(fname, img)
```

**流程說明：**
1. **初始化** → combination_image 複製 base_image
2. **迭代優化** → 每次迭代調整像素值，最小化損失
3. **損失演變** → Loss = 內容損失 + 風格損失 + 噪聲損失
4. **梯度下降** → 優化器根據梯度更新像素
5. **結果** → 4000 次迭代後得到風格轉移的圖像

**權重設置的含義：**
```python
content_weight = 2.5e-8   # 小 → 保留少量內容，主要改風格
style_weight = 1e-6       # 小 → 風格應用適度，不過度
total_variation_weight = 1e-6  # 很小 → 只做輔助平滑
```

---

## DeepDream

使用 InceptionV3 網絡產生夢幻式的視覺效果。通過最大化特定層的激活值來生成奇異的圖像。

**做法：**
1. **梯度上升** - 與風格轉移的梯度下降相反，最大化神經網絡對特定特徵的激活
2. **多尺度處理（Octave）** - 先在低解析度生成，再逐步放大到高解析度，保留細節
3. **特徵層選擇** - 選擇不同層級（mixed4~mixed7）的特徵組合，產生不同風格的夢幻效果
4. **細節保留** - 在放大過程中加回原始圖像的丟失細節，保持整體結構

簡單說：就是讓神經網絡「自由發揮」，把它認識到的圖案和特徵放大、強化，產生超現實的視覺效果。
