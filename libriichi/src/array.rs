/// このファイルは、固定列数の2次元配列を効率的に扱うための`Simple2DArray`構造体を定義しています。
/// 内部的には1次元のVecを使用し、行と列のインデックスから適切な位置を計算することで
/// 2次元配列のような操作を提供します。主な用途は、麻雀ゲームの状態表現や計算において
/// 効率的なメモリレイアウトと高速なアクセスが必要な場面での使用です。

/// 固定列数の2次元配列を表す構造体
/// const generic `COLS`で列数を型レベルで固定し、行数は動的に指定可能
pub struct Simple2DArray<const COLS: usize, T> {
    /// 実際のデータを格納する1次元配列
    /// 2次元の(row, col)は row * COLS + col でインデックスに変換される
    arr: Vec<T>,
}

/// Simple2DArrayの実装ブロック
/// T型はClone, Copy, Defaultトレイトを実装している必要がある
impl<const COLS: usize, T> Simple2DArray<COLS, T>
where
    T: Clone + Copy + Default,
{
    /// 指定された行数で新しい2次元配列を作成する
    #[inline]  // コンパイラに対してインライン化を推奨するヒント
    pub fn new(rows: usize) -> Self {
        // rows * COLS個の要素を持つVecを作成し、各要素をT::default()で初期化
        let arr = vec![Default::default(); rows * COLS];
        // 作成したVecを持つSelf構造体を返す
        Self { arr }
    }

    /// 指定された行と列の要素を取得する
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> T {
        // 2次元のインデックス(row, col)を1次元のインデックスに変換してアクセス
        // row * COLS で行の開始位置を計算し、colを加えて最終位置を決定
        self.arr[row * COLS + col]
    }

    /// 配列の行数を返す
    #[inline]
    pub const fn rows(&self) -> usize {
        // 全要素数を列数で割ることで行数を計算
        // constキーワードによりコンパイル時に評価可能
        self.arr.len() / COLS
    }

    /// 指定された行のすべての要素を同じ値で埋める
    /// - - - - -  <- 他の行は変更されない
    /// x x x x x  <- この行全体がvalueで埋められる
    /// - - - - -  <- 他の行は変更されない
    #[inline]
    pub fn fill(&mut self, row: usize, value: T) {
        // fill_rowsメソッドを使用して1行分を埋める
        self.fill_rows(row, 1, value);
    }

    /// 指定された行から複数行にわたってすべての要素を同じ値で埋める
    /// - - - - -  <- 他の行は変更されない
    /// x x x x x  <- rowから
    /// x x x x x  <- n_rows行分がvalueで埋められる
    #[inline]
    pub fn fill_rows(&mut self, row: usize, n_rows: usize, value: T) {
        // row * COLSで開始インデックス、(row + n_rows) * COLSで終了インデックスを計算
        // スライスのfillメソッドを使用して指定範囲をvalueで埋める
        self.arr[row * COLS..(row + n_rows) * COLS].fill(value);
    }

    /// 指定された位置の単一要素に値を設定する
    /// - - - - -  <- 他の要素は変更されない
    /// - - x - -  <- (row, col)の位置のみ変更
    /// - - - - -  <- 他の要素は変更されない
    #[inline]
    pub fn assign(&mut self, row: usize, col: usize, value: T) {
        // 2次元のインデックスを1次元に変換して値を代入
        self.arr[row * COLS + col] = value;
    }

    /// 指定された列の複数行にわたって同じ値を設定する
    /// - - x - -  <- col列のrowから
    /// - - x - -  <- n_rows行分が
    /// - - x - -  <- valueで埋められる
    #[inline]
    pub fn assign_rows(&mut self, row: usize, col: usize, n_rows: usize, value: T) {
        // n_rows回ループして各行の同じ列に値を設定
        for n in 0..n_rows {
            // (row + n)で現在の行、colで列を指定して値を代入
            self.arr[(row + n) * COLS + col] = value;
        }
    }

    /// 内部のVecデータをndarrayのArray2形式に変換する
    /// このメソッドは所有権を取得するため、selfを消費する
    #[inline]
    pub fn build(self) -> ndarray::Array2<T> {
        // (行数, 列数)のタプルで形状を定義
        let shape = (self.rows(), COLS);
        // 1次元のVecを指定された形状の2次元配列に変換
        // unwrap()は形状とVecのサイズが一致することが保証されているため安全
        ndarray::Array2::from_shape_vec(shape, self.arr).unwrap()
    }
}

/// テストモジュール
/// Simple2DArrayの各メソッドが正しく動作することを検証する
#[cfg(test)]  // テストビルド時のみこのモジュールをコンパイルする
mod test {
    use super::*;  // 親モジュールのすべての要素をインポート
    use ndarray::arr2;  // ndarray crateのarr2マクロをインポート（2次元配列を簡潔に作成するため）

    /// Simple2DArrayの変更操作（fill, fill_rows, assign, assign_rows）をテストする
    #[test]
    fn mutate() {
        // fill()メソッドのテスト：1行全体を埋める
        let mut arr = Simple2DArray::<2, i32>::new(4);  // 4行2列のi32型配列を作成
        arr.fill(1, 3);  // インデックス1の行（2行目）を3で埋める
        // 期待される結果：2行目のみが[3, 3]になり、他の行は初期値[0, 0]のまま
        assert_eq!(arr.build(), arr2(&[[0, 0], [3, 3], [0, 0], [0, 0]]));

        // fill_rows()メソッドのテスト：複数行を埋める
        let mut arr = Simple2DArray::<2, i32>::new(4);  // 新しい4行2列の配列を作成
        arr.fill_rows(1, 2, 3);  // インデックス1から2行分（2行目と3行目）を3で埋める
        // 期待される結果：2行目と3行目が[3, 3]になり、1行目と4行目は[0, 0]のまま
        assert_eq!(arr.build(), arr2(&[[0, 0], [3, 3], [3, 3], [0, 0]]));

        // assign()メソッドのテスト：単一要素を設定
        let mut arr = Simple2DArray::<2, i32>::new(4);  // 新しい4行2列の配列を作成
        arr.assign(1, 1, 3);  // (1, 1)の位置（2行目の2列目）に3を設定
        // 期待される結果：(1, 1)のみが3になり、他はすべて0のまま
        assert_eq!(arr.build(), arr2(&[[0, 0], [0, 3], [0, 0], [0, 0]]));

        // assign_rows()メソッドのテスト：同じ列の複数行に値を設定
        let mut arr = Simple2DArray::<2, i32>::new(4);  // 新しい4行2列の配列を作成
        arr.assign_rows(1, 1, 2, 3);  // 列インデックス1（2列目）の、行インデックス1から2行分に3を設定
        // 期待される結果：2行目と3行目の2列目が3になり、他は0のまま
        assert_eq!(arr.build(), arr2(&[[0, 0], [0, 3], [0, 3], [0, 0]]));
    }
}
