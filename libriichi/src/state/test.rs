//! PlayerStateのテストモジュール
//! 
//! このファイルは、麻雀のプレイヤー状態を管理する`PlayerState`構造体のテスト実装を含んでいます。
//! 主な機能:
//! - プレイヤー状態の更新と検証のためのテスト用ヘルパーメソッド
//! - 待ち牌判定、鳴き可能判定、フリテン判定などの各種ゲームロジックのテスト
//! - ドラ数計算、順位判定、和了判定などの得点計算系のテスト
//! - 実際のゲームログを使用した複雑なシナリオのテスト
//!
//! 各テストは、麻雀の特定のルールや状況を検証し、PlayerStateの実装が正しく動作することを保証します。

// PlayerStateとActionCandidate（行動候補）構造体を親モジュールからインポート
use super::{ActionCandidate, PlayerState};
// シャンテン数（和了までの最短手数）計算アルゴリズムをインポート
use crate::algo::shanten;
// サポートされている最大バージョン番号の定数をインポート
use crate::consts::MAX_VERSION;
// 手牌を扱うためのユーティリティ関数（文字列から手牌への変換など）をインポート
use crate::hand::{hand, hand_with_aka, tile37_to_vec};
// MJAI形式のイベント（ツモ、打牌など）を表す列挙型をインポート
use crate::mjai::Event;
// 牌を扱うためのマクロ（matches_tu8: 牌の一致判定、must_tile: インデックスから牌への変換、t: 牌リテラル、tuz: 牌のインデックス）をインポート
use crate::{matches_tu8, must_tile, t, tuz};
// メモリ操作のためのstd::memモジュール（mem::replaceで使用）をインポート
use std::mem;

// PlayerStateにテスト用のメソッドを実装
impl PlayerState {
    // テスト用のイベント更新メソッド
    // 通常のupdate()メソッドをラップし、更新後に必ず検証を実行する
    fn test_update(&mut self, event: &Event) -> ActionCandidate {
        // イベントを適用してプレイヤー状態を更新し、可能な行動を取得
        let cans = self.update(event).unwrap();
        // 状態の整合性を検証（シャンテン数、門前状態、ドラ数などが正しいか確認）
        self.validate();
        // 可能な行動を返す
        cans
    }

    // JSON形式のMJAIイベントでテストする更新メソッド
    // JSON文字列を受け取り、パースして更新を実行
    fn test_update_json(&mut self, mjai_json: &str) -> ActionCandidate {
        // JSON文字列からイベントをパースして更新
        let cans = self.update_json(mjai_json).unwrap();
        // 状態の整合性を検証
        self.validate();
        // 可能な行動を返す
        cans
    }

    // ゲームログから特定プレイヤーの状態を再現するメソッド
    // 複数行のMJAI JSONログを受け取り、順番に適用してゲーム状態を構築
    fn from_log(player_id: u8, log: &str) -> Self {
        // 指定されたプレイヤーIDで新しいPlayerStateを作成
        let mut ps = Self::new(player_id);
        // ログを改行で分割し、各行（JSONイベント）を順番に処理
        for line in log.trim().split('\n') {
            // 各イベントを適用（空行はスキップされる）
            ps.test_update_json(line);
        }
        // 構築されたゲーム状態を返す
        ps
    }

    // 手牌（手持ちの牌と副露）に含まれるドラの総数を計算
    fn num_doras_in_hand(&self) -> u8 {
        // 手牌の各牌について、その枚数とドラ係数を掛けて合計
        self.tehai
            .iter() // 手牌の各種類の枚数をイテレート
            .zip(self.dora_factor.iter()) // ドラ係数（その牌が何枚分のドラになるか）とペアにする
            .map(|(&count, &f)| count * f) // 枚数×ドラ係数でドラ数を計算
            // 手牌中の赤牌の数を追加（赤5萬、赤5筒、赤5索）
            .chain(self.akas_in_hand.iter().map(|&b| b as u8))
            // 副露（ポン、チー、明槓）中のドラを計算
            .chain(
                self.fuuro_overview[0] // プレイヤー0（自分）の副露
                    .iter() // 各副露をイテレート
                    .flatten() // Option<Tile>をTileに展開
                    .map(|t| self.dora_factor[t.deaka().as_usize()] + t.is_aka() as u8), // 通常ドラ＋赤ドラ
            )
            // 暗槓中のドラを計算（暗槓は4枚組なので×4）
            .chain(self.ankan_overview[0].iter().map(|t| {
                self.dora_factor[t.deaka().as_usize()] * 4  // 通常ドラ×4枚
                    + matches_tu8!(t.as_u8(), 5m | 5p | 5s) as u8 // 5の牌なら赤牌1枚分を追加
            }))
            .sum() // 全てのドラ数を合計
    }

    // PlayerStateの内部状態が整合性を保っているか検証
    fn validate(&self) {
        // リアルタイムシャンテン計算と通常のシャンテン計算の結果が一致するか確認
        assert_eq!(
            self.real_time_shanten(), // キャッシュされたシャンテン数
            shanten::calc_all(&self.tehai, self.tehai_len_div3), // 手牌から計算したシャンテン数
        );
        // 門前（メンゼン）フラグが副露の有無と一致するか確認
        assert_eq!(
            self.is_menzen, // 門前フラグ
            self.chis.is_empty() && self.pons.is_empty() && self.minkans.is_empty() // チー・ポン・明槓が無いこと
        );
        // 記録されているドラ数が実際の計算結果と一致するか確認
        assert_eq!(self.doras_owned[0], self.num_doras_in_hand());
        // 何らかの行動が可能な場合、観測データのエンコードが正しく動作するか確認
        if self.last_cans.can_act() {
            // 全てのサポートバージョンでエンコードをテスト
            for version in 1..=MAX_VERSION {
                // 通常のエンコード
                let _encoded = self.encode_obs(version, false);
                // カン（槓）が可能な場合は、カン用のエンコードもテスト
                if self.last_cans.can_kakan || self.last_cans.can_ankan {
                    let _encoded = self.encode_obs(version, true);
                }
            }
        }
    }
}

// 待ち牌判定のテスト
// 異なる手牌パターンで正しく待ち牌を判定できるか検証
#[test]
fn waits() {
    // テストケース1: 通常の待ち牌パターン
    let mut ps = PlayerState {
        // 456m 78999p 789s 77z = 4面子 + 1雀頭でテンパイ
        tehai: hand("456m 78999p 789s 77z").unwrap(),
        tehai_len_div3: 4, // 手牌枚数÷3 = 13÷3 = 4 (余り1)
        ..Default::default() // その他のフィールドはデフォルト値
    };
    // 待ち牌とフリテンの更新
    ps.update_waits_and_furiten();
    // 期待される待ち牌: 6筒、9筒・中(C)
    let expected = t![6p, 9p, C];
    // 全ての牌種について待ち牌かどうか確認
    for (idx, &b) in ps.waits.iter().enumerate() {
        if expected.contains(&must_tile!(idx)) {
            assert!(b); // 期待される待ち牌はtrue
        } else {
            assert!(!b); // それ以外はfalse
        }
    }

    // テストケース2: 多面張（複数の待ち牌）のパターン
    let mut ps = PlayerState {
        // 2344445666678s = 13枚のソウズで多面待ち
        tehai: hand("2344445666678s").unwrap(),
        tehai_len_div3: 4, // 手牌枚数÷3 = 13÷3 = 4 (余り1)
        ..Default::default()
    };
    // 待ち牌とフリテンの更新
    ps.update_waits_and_furiten();
    // 期待される待ち牌: 1s, 2s, 3s, 5s, 7s, 8s, 9s (清一色多面張)
    let expected = t![1s, 2s, 3s, 5s, 7s, 8s, 9s];
    // 全ての牌種について待ち牌かどうか確認
    for (idx, &b) in ps.waits.iter().enumerate() {
        if expected.contains(&must_tile!(idx)) {
            assert!(b); // 期待される待ち牌はtrue
        } else {
            assert!(!b); // それ以外はfalse
        }
    }
}

// チー（順子の鳴き）判定のテスト
// 様々な手牌と捨て牌の組み合わせで、正しくチー可能判定ができるか検証
#[test]
fn can_chi() {
    // プレイヤー0（親）で新しいPlayerStateを作成
    let mut ps = PlayerState::new(0);
    
    // テストケース1: 1111234mの手牌
    ps.tehai = hand("1111234m").unwrap();
    // 1mが捨てられた場合 -> チー不可（4枚とも持っているので組み合わせがない）
    ps.set_can_chi_from_tile(t!(1m));
    assert!(matches!(
        ps.last_cans,
        ActionCandidate {
            can_chi_high: false,  // 上チー（例: 6-7でと8でチー）不可
            can_chi_mid: false,   // 中チー（例: 5-7でと6でチー）不可
            can_chi_low: false,   // 下チー（例: 7-8でと6でチー）不可
            ..
        },
    ));
    // 4mが捨てられた場合 -> チー不可（234が既にあるため）
    ps.set_can_chi_from_tile(t!(4m));
    assert!(matches!(
        ps.last_cans,
        ActionCandidate {
            can_chi_high: false,
            can_chi_mid: false,
            can_chi_low: false,
            ..
        },
    ));
    // 2mが捨てられた場合 -> チー可能（134、234の両方可能）
    ps.set_can_chi_from_tile(t!(2m));
    assert!(matches!(
        ps.last_cans,
        ActionCandidate {
            can_chi_high: false,  // 123の上チーはできない
            can_chi_mid: true,    // 123の中チー可能 (1-3で2でチー)
            can_chi_low: true,    // 234の下チー可能 (3-4で2でチー)
            ..
        },
    ));

    // テストケース2: 6666789999pの手牌
    ps.tehai = hand("6666789999p").unwrap();
    // 5pが捨てられた場合 -> 567の下チーのみ可能
    ps.set_can_chi_from_tile(t!(5p));
    assert!(matches!(
        ps.last_cans,
        ActionCandidate {
            can_chi_high: false,
            can_chi_mid: false,
            can_chi_low: true,    // 567の下チーのみ可能
            ..
        },
    ));
    // 7pが捨てられた場合 -> 678、789の両方可能
    ps.set_can_chi_from_tile(t!(7p));
    assert!(matches!(
        ps.last_cans,
        ActionCandidate {
            can_chi_high: false,
            can_chi_mid: true,    // 678の中チー可能
            can_chi_low: true,    // 789の下チー可能
            ..
        },
    ));
    // 8pが捨てられた場合 -> 678、789の両方可能
    ps.set_can_chi_from_tile(t!(8p));
    assert!(matches!(
        ps.last_cans,
        ActionCandidate {
            can_chi_high: true,   // 678の上チー可能
            can_chi_mid: true,    // 789の中チー可能
            can_chi_low: false,
            ..
        },
    ));

    // テストケース3: 4556sの手牌（重複あり）
    ps.tehai = hand("4556s").unwrap();
    // 3sが捨てられた場合 -> 345の下チーのみ可能
    ps.set_can_chi_from_tile(t!(3s));
    assert!(matches!(
        ps.last_cans,
        ActionCandidate {
            can_chi_high: false,
            can_chi_mid: false,
            can_chi_low: true,    // 345の下チー可能
            ..
        },
    ));
    // 4sが捨てられた場合 -> 456の下チーのみ可能（1枚持っているため）
    ps.set_can_chi_from_tile(t!(4s));
    assert!(matches!(
        ps.last_cans,
        ActionCandidate {
            can_chi_high: false,
            can_chi_mid: false,
            can_chi_low: true,    // 456の下チー可能
            ..
        },
    ));
    // 5sが捨てられた場合 -> チー不可（2枚持っているため）
    ps.set_can_chi_from_tile(t!(5s));
    assert!(matches!(
        ps.last_cans,
        ActionCandidate {
            can_chi_high: false,
            can_chi_mid: false,
            can_chi_low: false,
            ..
        },
    ));
    // 6sが捨てられた場合 -> 456の上チーのみ可能
    ps.set_can_chi_from_tile(t!(6s));
    assert!(matches!(
        ps.last_cans,
        ActionCandidate {
            can_chi_high: true,   // 456の上チー可能
            can_chi_mid: false,
            can_chi_low: false,
            ..
        },
    ));
    // 7sが捨てられた場合 -> 567の上チーのみ可能
    ps.set_can_chi_from_tile(t!(7s));
    assert!(matches!(
        ps.last_cans,
        ActionCandidate {
            can_chi_high: true,   // 567の上チー可能
            can_chi_mid: false,
            can_chi_low: false,
            ..
        },
    ));
}

// フリテン（振な聴牌）のテスト
// 様々なフリテン状態の判定と、リーチ後のフリテン処理を検証
#[test]
fn furiten() {
    // プレイヤー0で新しいPlayerStateを作成
    let mut ps = PlayerState::new(0);
    // 東場南家でゲーム開始
    ps.test_update(&Event::StartKyoku {
        bakaze: t!(E), // 場風: 東
        kyoku: 1,      // 局数: 1 (東一局)
        honba: 0,      // 本場数: 0
        kyotaku: 0,    // 供托数: 0
        oya: 0,        // 親: プレイヤー0
        scores: [25000; 4], // 各プレイヤーの点数: 25000点ずつ
        dora_marker: t!(3p), // ドラ表示牌: 3筒（ドラは4筒）
        tehais: [
            // プレイヤー0の手牌: 23406m 456789p 58s
            tile37_to_vec(&hand_with_aka("23406m 456789p 58s").unwrap())
                .try_into()
                .unwrap(),
            // 他のプレイヤーは不明牌
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
        ],
    });
    // プレイヤー0が8sをツモ
    ps.test_update(&Event::Tsumo {
        actor: 0,
        pai: t!(8s),
    });
    assert!(ps.shanten == 1); // 1シャンテン
    assert!(ps.waits.iter().all(|&b| !b)); // まだテンパイではないので待ち牌なし
    // 5sを打牌してテンパイに
    ps.test_update(&Event::Dahai {
        actor: 0,
        pai: t!(5s),
        tsumogiri: false, // 手出し
    });
    assert!(ps.shanten == 0); // 0シャンテン（テンパイ）
    assert!(ps.waits[tuz!(1m)] && ps.waits[tuz!(4m)] && ps.waits[tuz!(7m)]); // 1m, 4m, 7m待ち
    assert!(!ps.at_furiten); // フリテンではない

    // 通常フリテンのテスト
    // プレイヤー1がツモと打牌
    ps.test_update(&Event::Tsumo {
        actor: 1,
        pai: t!(?), // 不明牌
    });
    // プレイヤー1が1mを打牌（プレイヤー0の待ち牌）
    let cans = ps.test_update(&Event::Dahai {
        actor: 1,
        pai: t!(1m),
        tsumogiri: false,
    });
    assert!(!ps.at_furiten); // まだフリテンではない
    assert!(cans.can_ron_agari); // ロン可能

    // プレイヤー2がツモ（プレイヤー0が1mを見逃したのでフリテンに）
    ps.test_update(&Event::Tsumo {
        actor: 2,
        pai: t!(?),
    });
    assert!(ps.at_furiten); // 見逃しによる同巡フリテン
    ps.test_update(&Event::Dahai {
        actor: 2,
        pai: t!(1s),
        tsumogiri: true,
    });

    // プレイヤー3がツモと打牌
    ps.test_update(&Event::Tsumo {
        actor: 3,
        pai: t!(?),
    });
    // プレイヤー3が1mを打牌（再びプレイヤー0の待ち牌）
    let cans = ps.test_update(&Event::Dahai {
        actor: 3,
        pai: t!(1m),
        tsumogiri: false,
    });
    assert!(ps.shanten == 0); // まだテンパイ
    assert!(ps.waits[tuz!(1m)] && ps.waits[tuz!(4m)] && ps.waits[tuz!(7m)]); // 待ち牌は変わらず
    assert!(ps.at_furiten); // フリテン中
    assert!(!cans.can_ron_agari); // フリテンなのでロン不可

    // フリテン解除のテスト
    // プレイヤー0がツモと打牌
    ps.test_update(&Event::Tsumo {
        actor: 0,
        pai: t!(3s),
    });
    assert!(ps.at_furiten); // まだフリテン中
    ps.test_update(&Event::Dahai {
        actor: 0,
        pai: t!(3s),
        tsumogiri: true, // ツモ切り
    });
    assert!(!ps.at_furiten); // 自分が打牌したのでフリテン解除

    // フリテン解除後のロン可否テスト
    // プレイヤー1のターン
    ps.test_update(&Event::Tsumo {
        actor: 1,
        pai: t!(?),
    });
    ps.test_update(&Event::Dahai {
        actor: 1,
        pai: t!(P), // 白を打牌
        tsumogiri: true,
    });

    // プレイヤー2のターン
    ps.test_update(&Event::Tsumo {
        actor: 2,
        pai: t!(?),
    });
    ps.test_update(&Event::Dahai {
        actor: 2,
        pai: t!(C), // 中を打牌
        tsumogiri: true,
    });
    
    // プレイヤー3のターン
    ps.test_update(&Event::Tsumo {
        actor: 3,
        pai: t!(?),
    });
    // プレイヤー3が再び1mを打牌
    let cans = ps.test_update(&Event::Dahai {
        actor: 3,
        pai: t!(1m),
        tsumogiri: false,
    });
    assert!(!ps.at_furiten); // フリテン解除されている
    assert!(cans.can_ron_agari); // ロン可能
    assert_eq!(ps.agari_points(true, &[]).unwrap().ron, 5800); // ロン点数: 5800点

    // リーチ後のフリテンテスト
    // プレイヤー0が北(N)をツモ
    let cans = ps.test_update(&Event::Tsumo {
        actor: 0,
        pai: t!(N),
    });
    assert!(cans.can_riichi); // リーチ可能
    // リーチ宣言
    ps.test_update(&Event::Reach { actor: 0 });
    // 北を打牌してリーチ
    ps.test_update(&Event::Dahai {
        actor: 0,
        pai: t!(N),
        tsumogiri: true,
    });
    // リーチ成立
    ps.test_update(&Event::ReachAccepted { actor: 0 });

    // 他プレイヤーのターンを3巡
    // プレイヤー1のターン
    ps.test_update(&Event::Tsumo {
        actor: 1,
        pai: t!(?),
    });
    ps.test_update(&Event::Dahai {
        actor: 1,
        pai: t!(9m), // 9萬を打牌
        tsumogiri: true,
    });
    
    // プレイヤー2のターン
    ps.test_update(&Event::Tsumo {
        actor: 2,
        pai: t!(?),
    });
    ps.test_update(&Event::Dahai {
        actor: 2,
        pai: t!(9m), // 9萬を打牌
        tsumogiri: true,
    });
    
    // プレイヤー3のターン
    ps.test_update(&Event::Tsumo {
        actor: 3,
        pai: t!(?),
    });
    ps.test_update(&Event::Dahai {
        actor: 3,
        pai: t!(9m), // 9萬を打牌
        tsumogiri: true,
    });

    // リーチ後のツモ和了見逃しによる永続フリテン
    // プレイヤー0が1mをツモ（待ち牌）
    let cans = ps.test_update(&Event::Tsumo {
        actor: 0,
        pai: t!(1m), // 待ち牌の1萬をツモ
    });
    assert!(ps.waits[tuz!(1m)] && ps.waits[tuz!(4m)] && ps.waits[tuz!(7m)]); // 待ち牌確認
    assert!(!ps.at_furiten); // ツモ時点ではフリテンではない
    assert!(cans.can_tsumo_agari); // ツモ和了可能
    // ツモ和了しないで1mを打牌
    ps.test_update(&Event::Dahai {
        actor: 0,
        pai: t!(1m),
        tsumogiri: true,
    });
    assert!(ps.at_furiten); // リーチ中に和了牌を捨てたので永続フリテン

    // 永続フリテン中のロン不可確認
    // プレイヤー1のターン
    ps.test_update(&Event::Tsumo {
        actor: 1,
        pai: t!(?),
    });
    ps.test_update(&Event::Dahai {
        actor: 1,
        pai: t!(4s), // 4索を打牌
        tsumogiri: true,
    });
    
    // プレイヤー2のターン
    ps.test_update(&Event::Tsumo {
        actor: 2,
        pai: t!(?),
    });
    ps.test_update(&Event::Dahai {
        actor: 2,
        pai: t!(4s), // 4索を打牌
        tsumogiri: true,
    });
    
    // プレイヤー3が7m（待ち牌）を打牌
    ps.test_update(&Event::Tsumo {
        actor: 3,
        pai: t!(?),
    });
    let cans = ps.test_update(&Event::Dahai {
        actor: 3,
        pai: t!(7m), // 待ち牌の7萬を打牌
        tsumogiri: true,
    });
    assert!(ps.waits[tuz!(1m)] && ps.waits[tuz!(4m)] && ps.waits[tuz!(7m)]); // 待ち牌は変わらず
    assert!(ps.at_furiten); // 永続フリテン中
    assert!(!cans.can_ron_agari); // フリテンなのでロン不可

    // 永続フリテンの継続確認
    // プレイヤー0が8mをツモして打牌
    ps.test_update(&Event::Tsumo {
        actor: 0,
        pai: t!(8m),
    });
    ps.test_update(&Event::Dahai {
        actor: 0,
        pai: t!(8m),
        tsumogiri: true,
    });
    assert!(ps.at_furiten); // 依然としてフリテン中（リーチ後の永続フリテンは解除されない）

    // 他プレイヤーのターン（永続フリテン中）
    // プレイヤー1のターン
    ps.test_update(&Event::Tsumo {
        actor: 1,
        pai: t!(?),
    });
    ps.test_update(&Event::Dahai {
        actor: 1,
        pai: t!(E), // 東を打牌
        tsumogiri: true,
    });
    
    // プレイヤー2が4m（待ち牌）を打牌
    ps.test_update(&Event::Tsumo {
        actor: 2,
        pai: t!(?),
    });
    let cans = ps.test_update(&Event::Dahai {
        actor: 2,
        pai: t!(4m), // 待ち牌の4萬を打牌
        tsumogiri: true,
    });
    assert!(ps.at_furiten); // まだフリテン中
    assert!(!cans.can_ron_agari); // ロン不可
    
    // プレイヤー3のターン
    ps.test_update(&Event::Tsumo {
        actor: 3,
        pai: t!(?),
    });
    ps.test_update(&Event::Dahai {
        actor: 3,
        pai: t!(E), // 東を打牌
        tsumogiri: true,
    });

    // フリテン中でもツモ和了は可能
    // プレイヤー0が4m（待ち牌）をツモ
    let cans = ps.test_update(&Event::Tsumo {
        actor: 0,
        pai: t!(4m), // 待ち牌の4萬をツモ
    });
    assert!(ps.waits[0] && ps.waits[3] && ps.waits[6]); // 待ち牌確認（インデックス0,3,6 = 1m,4m,7m）
    assert!(ps.at_furiten); // フリテン中
    assert!(cans.can_tsumo_agari); // フリテンでもツモ和了は可能
    assert_eq!(ps.agari_points(false, &[t!(3m)]).unwrap().tsumo_ko, 6000); // ツモ点数: 6000点（ドラ3萬付き）
}

// カン（槓）後のドラ数計算テスト
// 暗槓、ポン、明槓などの後にドラ数が正しく更新されるか検証
#[test]
fn dora_count_after_kan() {
    // プレイヤー0で新しいPlayerStateを作成
    let mut ps = PlayerState::new(0);
    // ゲーム開始
    ps.test_update(&Event::StartKyoku {
        bakaze: t!(E), // 場風: 東
        kyoku: 1,      // 局数: 1
        honba: 0,      // 本場数: 0
        kyotaku: 0,    // 供托数: 0
        oya: 0,        // 親: プレイヤー0
        scores: [25000; 4], // 各プレイヤーの点数
        dora_marker: t!(N), // ドラ表示牌: 北（ドラは1索）
        tehais: [
            // プレイヤー0の手牌: 1111s 123456p 112z（4枚の1索を持っている）
            tile37_to_vec(&hand_with_aka("1111s 123456p 112z").unwrap())
                .try_into()
                .unwrap(),
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
        ],
    });
    // 8sをツモ
    ps.test_update(&Event::Tsumo {
        actor: 0,
        pai: t!(8s),
    });
    assert_eq!(ps.doras_owned[0], 2); // ドラ数: 2（現在のドラは1索、4枚中2枚持っている）

    // 暗槓と新ドラのテスト
    // 1索を4枚で暗槓
    ps.test_update(&Event::Ankan {
        actor: 0,
        consumed: [t!(1s); 4], // 1索4枚で暗槓
    });
    // 新ドラ表示牌: 9索（新ドラは1索）
    ps.test_update(&Event::Dora {
        dora_marker: t!(9s),
    });
    // 嶺上ツモ: 赤5筒
    ps.test_update(&Event::Tsumo {
        actor: 0,
        pai: t!(5pr), // 赤5筒をツモ
    });
    assert_eq!(ps.doras_owned[0], 7); // ドラ数: 7（暗槓の1索4枚 + 旧ドラ1索2枚 + 赤5筒1枚）
    // 東を打牌
    ps.test_update(&Event::Dahai {
        actor: 0,
        pai: t!(E), // 東を打牌
        tsumogiri: true,
    });
    assert_eq!(ps.doras_owned[0], 6); // ドラ数: 6（東はドラではないので変わらず）

    // ポン後のドラ数テスト
    // プレイヤー1が5pを打牌
    ps.test_update(&Event::Tsumo {
        actor: 1,
        pai: t!(?),
    });
    ps.test_update(&Event::Dahai {
        actor: 1,
        pai: t!(5p), // 5筒を打牌
        tsumogiri: true,
    });

    // プレイヤー0が5pをポン（赤5筒を5筒を消費）
    ps.test_update(&Event::Pon {
        actor: 0,
        target: 1,
        pai: t!(5p),
        consumed: t![5pr, 5p], // 赤5筒と5筒でポン
    });
    assert_eq!(ps.doras_owned[0], 6); // ドラ数: 6（持っていた赤5筒をポンに使ったので-1）
    // 東を打牌
    ps.test_update(&Event::Dahai {
        actor: 0,
        pai: t!(E), // 東を打牌
        tsumogiri: false,
    });
    assert_eq!(ps.doras_owned[0], 5); // ドラ数: 5（旧ドラ1索2枚が1枚に減った）

    // 他プレイヤーの暗槓後のドラ数確認
    // プレイヤー1と2のターン
    ps.test_update(&Event::Tsumo {
        actor: 1,
        pai: t!(?),
    });
    ps.test_update(&Event::Dahai {
        actor: 1,
        pai: t!(P), // 白を打牌
        tsumogiri: true,
    });
    ps.test_update(&Event::Tsumo {
        actor: 2,
        pai: t!(?),
    });
    ps.test_update(&Event::Dahai {
        actor: 2,
        pai: t!(P), // 白を打牌
        tsumogiri: true,
    });

    // プレイヤー3が1mを暗槓
    ps.test_update(&Event::Tsumo {
        actor: 3,
        pai: t!(?),
    });
    ps.test_update(&Event::Ankan {
        actor: 3,
        consumed: [t!(1m); 4], // 1萬4枚で暗槓
    });
    // 新ドラ表示牌: 4筒（新ドラは5筒）
    ps.test_update(&Event::Dora {
        dora_marker: t!(4p),
    });
    assert_eq!(ps.doras_owned[0], 8); // ドラ数: 8（暗槓1索4枚 + 旧ドラ1枚 + ポン5筒3枚）1枚は赤5筒））
}

#[test]
fn rule_based_agari_all_last_minogashi() {
    let log = r#"
        {"type":"start_kyoku","bakaze":"S","dora_marker":"5m","kyoku":4,"honba":0,"kyotaku":0,"oya":3,"scores":[35300,3000,38400,23300],"tehais":[["4m","5mr","8m","1p","3p","3p","5p","2s","5sr","9s","W","P","P"],["2m","3m","5m","7m","7p","9p","4s","5s","5s","6s","7s","7s","E"],["3m","5m","6m","2p","6p","9p","1s","5s","8s","9s","S","S","C"],["1m","4m","3p","4p","5pr","7p","1s","2s","7s","8s","W","N","P"]]}
        {"type":"tsumo","actor":3,"pai":"F"}
        {"type":"dahai","actor":3,"pai":"1m","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"5p"}
        {"type":"dahai","actor":0,"pai":"W","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"9m"}
        {"type":"dahai","actor":1,"pai":"E","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"N"}
        {"type":"dahai","actor":2,"pai":"9p","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"2p"}
        {"type":"dahai","actor":3,"pai":"N","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"6m"}
        {"type":"dahai","actor":0,"pai":"9s","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"7m"}
        {"type":"dahai","actor":1,"pai":"9m","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"3s"}
        {"type":"dahai","actor":2,"pai":"2p","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"4s"}
        {"type":"dahai","actor":3,"pai":"W","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"1m"}
        {"type":"dahai","actor":0,"pai":"1m","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"9m"}
        {"type":"dahai","actor":1,"pai":"9m","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"3m"}
        {"type":"dahai","actor":2,"pai":"N","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"2s"}
        {"type":"dahai","actor":3,"pai":"F","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"2m"}
        {"type":"dahai","actor":0,"pai":"2s","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"1m"}
        {"type":"dahai","actor":1,"pai":"5m","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"3p"}
        {"type":"dahai","actor":2,"pai":"3p","tsumogiri":true}
        {"type":"pon","actor":0,"target":2,"pai":"3p","consumed":["3p","3p"]}
        {"type":"dahai","actor":0,"pai":"2m","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"6p"}
        {"type":"dahai","actor":1,"pai":"9p","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"6s"}
        {"type":"dahai","actor":2,"pai":"C","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"7p"}
        {"type":"dahai","actor":3,"pai":"P","tsumogiri":false}
        {"type":"pon","actor":0,"target":3,"pai":"P","consumed":["P","P"]}
        {"type":"dahai","actor":0,"pai":"1p","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"7s"}
        {"type":"dahai","actor":1,"pai":"5s","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"3s"}
        {"type":"dahai","actor":2,"pai":"9s","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"2m"}
        {"type":"dahai","actor":3,"pai":"1s","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"1p"}
        {"type":"dahai","actor":0,"pai":"1p","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"7m"}
        {"type":"dahai","actor":1,"pai":"4s","tsumogiri":false}
        {"type":"chi","actor":2,"target":1,"pai":"4s","consumed":["5s","6s"]}
        {"type":"dahai","actor":2,"pai":"6p","tsumogiri":false}
        {"type":"chi","actor":3,"target":2,"pai":"6p","consumed":["5pr","7p"]}
        {"type":"dahai","actor":3,"pai":"7p","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"1s"}
        {"type":"dahai","actor":0,"pai":"1s","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"1s"}
        {"type":"reach","actor":1}
        {"type":"dahai","actor":1,"pai":"1s","tsumogiri":true}
        {"type":"reach_accepted","actor":1}
        {"type":"tsumo","actor":2,"pai":"9s"}
        {"type":"dahai","actor":2,"pai":"8s","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"4p"}
        {"type":"dahai","actor":3,"pai":"4p","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"4m"}
        {"type":"dahai","actor":0,"pai":"4m","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"1p"}
        {"type":"dahai","actor":1,"pai":"1p","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"8m"}
        {"type":"dahai","actor":2,"pai":"8m","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"C"}
        {"type":"dahai","actor":3,"pai":"C","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"2s"}
        {"type":"dahai","actor":0,"pai":"2s","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"8p"}
    "#;
    let mut ps = PlayerState::from_log(1, log);

    assert!(ps.last_cans.can_tsumo_agari);
    let should_hora = ps.rule_based_agari();
    assert!(!should_hora);

    let orig_scores = mem::replace(&mut ps.scores, [9000, 30000, 30000, 30000]);
    let should_hora = ps.rule_based_agari();
    assert!(should_hora);
    ps.scores = orig_scores;

    ps.add_dora_indicator(t!(5m)).unwrap();
    let should_hora = ps.rule_based_agari();
    assert!(should_hora);

    let log = r#"
        {"type":"start_kyoku","bakaze":"S","dora_marker":"3s","kyoku":4,"honba":1,"kyotaku":0,"oya":3,"scores":[39000,25000,16900,19100],"tehais":[["1m","2m","3m","5mr","6m","8m","2p","2p","5pr","7s","8s","S","S"],["7m","9m","9m","6p","7p","1s","1s","3s","4s","6s","6s","S","P"],["3m","4m","5m","7m","4p","5p","5p","6p","8p","9p","5sr","5s","F"],["1m","2m","2m","6m","8m","1p","9p","3s","5s","6s","7s","E","W"]]}
        {"type":"tsumo","actor":3,"pai":"N"}
        {"type":"dahai","actor":3,"pai":"9p","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"1s"}
        {"type":"dahai","actor":0,"pai":"5pr","tsumogiri":false}
        {"type":"pon","actor":2,"target":0,"pai":"5pr","consumed":["5p","5p"]}
        {"type":"dahai","actor":2,"pai":"9p","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"7m"}
        {"type":"dahai","actor":3,"pai":"1p","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"C"}
        {"type":"dahai","actor":0,"pai":"8m","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"7m"}
        {"type":"dahai","actor":1,"pai":"6p","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"9s"}
        {"type":"dahai","actor":2,"pai":"9s","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"F"}
        {"type":"dahai","actor":3,"pai":"W","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"6m"}
        {"type":"dahai","actor":0,"pai":"1s","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"2m"}
        {"type":"dahai","actor":1,"pai":"7p","tsumogiri":false}
        {"type":"chi","actor":2,"target":1,"pai":"7p","consumed":["6p","8p"]}
        {"type":"dahai","actor":2,"pai":"F","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"4p"}
        {"type":"dahai","actor":3,"pai":"N","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"W"}
        {"type":"dahai","actor":0,"pai":"W","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"6m"}
        {"type":"dahai","actor":1,"pai":"2m","tsumogiri":false}
        {"type":"pon","actor":3,"target":1,"pai":"2m","consumed":["2m","2m"]}
        {"type":"dahai","actor":3,"pai":"F","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"4s"}
        {"type":"dahai","actor":0,"pai":"4s","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"1s"}
        {"type":"dahai","actor":1,"pai":"P","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"8s"}
        {"type":"dahai","actor":2,"pai":"8s","tsumogiri":true}
        {"type":"chi","actor":3,"target":2,"pai":"8s","consumed":["6s","7s"]}
        {"type":"dahai","actor":3,"pai":"1m","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"3p"}
        {"type":"dahai","actor":0,"pai":"C","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"6p"}
        {"type":"dahai","actor":1,"pai":"6p","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"4s"}
        {"type":"dahai","actor":2,"pai":"4p","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"N"}
        {"type":"dahai","actor":3,"pai":"N","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"3s"}
        {"type":"dahai","actor":0,"pai":"3s","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"5s"}
        {"type":"dahai","actor":1,"pai":"S","tsumogiri":false}
        {"type":"pon","actor":0,"target":1,"pai":"S","consumed":["S","S"]}
        {"type":"dahai","actor":0,"pai":"3p","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"3m"}
        {"type":"dahai","actor":1,"pai":"3m","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"4p"}
        {"type":"dahai","actor":2,"pai":"4p","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"P"}
        {"type":"dahai","actor":3,"pai":"P","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"8p"}
        {"type":"dahai","actor":0,"pai":"8p","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"4p"}
        {"type":"dahai","actor":1,"pai":"4p","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"E"}
        {"type":"dahai","actor":2,"pai":"E","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"C"}
        {"type":"dahai","actor":3,"pai":"4p","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"7p"}
        {"type":"dahai","actor":0,"pai":"7p","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"8p"}
        {"type":"dahai","actor":1,"pai":"8p","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"S"}
        {"type":"dahai","actor":2,"pai":"S","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"N"}
        {"type":"dahai","actor":3,"pai":"N","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"2s"}
        {"type":"dahai","actor":0,"pai":"2s","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"8s"}
        {"type":"dahai","actor":1,"pai":"8s","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"E"}
        {"type":"dahai","actor":2,"pai":"E","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"6s"}
        {"type":"dahai","actor":3,"pai":"E","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"9m"}
        {"type":"dahai","actor":0,"pai":"9m","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"F"}
        {"type":"dahai","actor":1,"pai":"F","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"C"}
        {"type":"dahai","actor":2,"pai":"C","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"E"}
        {"type":"dahai","actor":3,"pai":"E","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"W"}
        {"type":"dahai","actor":0,"pai":"W","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"P"}
        {"type":"dahai","actor":1,"pai":"P","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"N"}
        {"type":"dahai","actor":2,"pai":"N","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"8m"}
        {"type":"dahai","actor":3,"pai":"C","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"P"}
        {"type":"dahai","actor":0,"pai":"P","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"4m"}
        {"type":"dahai","actor":1,"pai":"9m","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"5m"}
        {"type":"dahai","actor":2,"pai":"4s","tsumogiri":false}
        {"type":"chi","actor":3,"target":2,"pai":"4s","consumed":["5s","6s"]}
        {"type":"dahai","actor":3,"pai":"3s","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"1m"}
        {"type":"dahai","actor":0,"pai":"1m","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"8s"}
        {"type":"dahai","actor":1,"pai":"9m","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"9s"}
        {"type":"dahai","actor":2,"pai":"9s","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"7s"}
        {"type":"dahai","actor":3,"pai":"7s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"7s"}
        {"type":"dahai","actor":0,"pai":"6m","tsumogiri":false}
    "#;
    let ps = PlayerState::from_log(2, log);
    assert!(ps.rule_based_agari());
}

// 順位計算のテスト
// 点数配列からプレイヤーの順位を正しく計算できるか検証
#[test]
fn get_rank() {
    // テストケース1: プレイヤー0が最下位
    let ps = PlayerState::new(0);
    let rank = ps.get_rank([20000, 25000, 25000, 30000]); // 点数: 20000, 25000, 25000, 30000
    assert_eq!(rank, 3); // 4位（インデックス3）

    // テストケース2: 全員同点でプレイヤー3の順位
    let ps = PlayerState::new(3);
    let rank = ps.get_rank([25000, 25000, 25000, 25000]); // 全員同点
    assert_eq!(rank, 3); // 同点の場合、プレイヤー3は4位（親優先）

    // テストケース3: プレイヤー1が2位
    let ps = PlayerState::new(1);
    let rank = ps.get_rank([25000, 30000, 20000, 25000]); // 点数: 25000, 30000, 20000, 25000
    assert_eq!(rank, 2); // 2位（インデックス2）

    // テストケース4: 同点が複数ある場合のプレイヤー1
    let ps = PlayerState::new(1);
    let rank = ps.get_rank([32000, 32000, 18000, 18000]); // 32000点2人、18000点2人
    assert_eq!(rank, 0); // 1位（プレイヤー0と同点だが、席順で負ける）

    // テストケース5: 同点処理の確認
    let ps = PlayerState::new(2);
    let rank = ps.get_rank([32000, 18000, 18000, 32000]); // プレイヤー2は18000点
    assert_eq!(rank, 1); // 3位（プレイヤー1と同点だが、席順で負ける）

    // テストケース6: 少ない点数でのテスト
    let ps = PlayerState::new(2);
    let rank = ps.get_rank([5, 2, 5, 3]); // 点数: 5, 2, 5, 3
    assert_eq!(rank, 1); // 2位（プレイヤー0と同点だが、席順で負ける）
}

// 手牌からの加槓（カカン）テスト
// ポンした牌を後から加槓する場合の処理を検証
#[test]
fn kakan_from_hand() {
    // 南二局のゲームログ
    let log = r#"
        {"type":"start_kyoku","bakaze":"S","dora_marker":"6m","kyoku":2,"honba":0,"kyotaku":0,"oya":1,"scores":[16100,36600,16800,30500],"tehais":[["5p","5s","1s","9m","9m","W","E","N","1p","F","9m","3p","6p"],["4s","9s","S","4s","1m","P","N","7s","F","2m","3s","2s","2s"],["6m","8p","8p","2p","8m","N","7p","C","1s","2p","N","9s","9p"],["2m","6s","7p","9s","2m","9s","6m","7s","8m","3m","S","5mr","C"]]}
        {"type":"tsumo","actor":1,"pai":"S"}
        {"type":"dahai","actor":1,"pai":"N","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"1s"}
        {"type":"dahai","actor":2,"pai":"9s","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"P"}
        {"type":"dahai","actor":3,"pai":"S","tsumogiri":false}
        {"type":"pon","actor":1,"target":3,"pai":"S","consumed":["S","S"]}
        {"type":"dahai","actor":1,"pai":"P","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"4p"}
        {"type":"dahai","actor":2,"pai":"C","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"5s"}
        {"type":"dahai","actor":3,"pai":"C","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"7m"}
        {"type":"dahai","actor":0,"pai":"E","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"P"}
        {"type":"dahai","actor":1,"pai":"1m","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"9p"}
        {"type":"dahai","actor":2,"pai":"6m","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"C"}
        {"type":"dahai","actor":3,"pai":"C","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"7p"}
        {"type":"dahai","actor":0,"pai":"W","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"5s"}
        {"type":"dahai","actor":1,"pai":"2m","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"5m"}
        {"type":"dahai","actor":2,"pai":"5m","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"1p"}
        {"type":"dahai","actor":3,"pai":"1p","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"4m"}
        {"type":"dahai","actor":0,"pai":"N","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"E"}
        {"type":"dahai","actor":1,"pai":"P","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"1s"}
        {"type":"dahai","actor":2,"pai":"8m","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"6p"}
        {"type":"dahai","actor":3,"pai":"8m","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"5p"}
        {"type":"dahai","actor":0,"pai":"1s","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"2s"}
        {"type":"dahai","actor":1,"pai":"E","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"5m"}
        {"type":"dahai","actor":2,"pai":"5m","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"3s"}
        {"type":"dahai","actor":3,"pai":"3s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"7p"}
        {"type":"dahai","actor":0,"pai":"F","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"E"}
        {"type":"dahai","actor":1,"pai":"E","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"W"}
        {"type":"dahai","actor":2,"pai":"W","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"7m"}
        {"type":"dahai","actor":3,"pai":"2m","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"5m"}
        {"type":"dahai","actor":0,"pai":"5s","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"S"}
        {"type":"dahai","actor":1,"pai":"F","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"6p"}
        {"type":"dahai","actor":2,"pai":"N","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"2p"}
        {"type":"dahai","actor":3,"pai":"2p","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"6p"}
        {"type":"dahai","actor":0,"pai":"3p","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"4m"}
        {"type":"dahai","actor":1,"pai":"4m","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"3s"}
        {"type":"dahai","actor":2,"pai":"N","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"8p"}
        {"type":"reach","actor":3}
        {"type":"dahai","actor":3,"pai":"P","tsumogiri":false}
        {"type":"reach_accepted","actor":3}
        {"type":"tsumo","actor":0,"pai":"W"}
        {"type":"dahai","actor":0,"pai":"1p","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"8s"}
        // プレイヤー1が南(S)を加槓（ポンしていた南に4枚目を加える）
        {"type":"kakan","actor":1,"pai":"S","consumed":["S","S","S"]}
        // 嶺上ツモ: 4索
        {"type":"tsumo","actor":1,"pai":"4s"}
    "#;
    // プレイヤー1の状態をログから再現
    let ps = PlayerState::from_log(1, log);

    // 加槓後にツモ和了可能か確認
    assert!(ps.last_cans.can_tsumo_agari);
}

#[test]
fn discard_candidates_with_unconditional_tenpai() {
    let log = r#"
        {"type":"start_kyoku","bakaze":"S","dora_marker":"2s","kyoku":3,"honba":0,"kyotaku":0,"oya":2,"scores":[25600,15600,21200,37600],"tehais":[["3m","3m","1p","6p","7p","9p","5sr","7s","8s","8s","E","E","W"],["4m","5mr","6m","1p","4p","5p","8p","3s","3s","4s","5s","S","P"],["1m","5m","7m","2p","9p","3s","5s","9s","S","W","N","P","C"],["1m","4m","6m","2p","3p","4p","6p","9p","2s","4s","7s","S","N"]]}
        {"type":"tsumo","actor":2,"pai":"C"}
        {"type":"dahai","actor":2,"pai":"N","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"2m"}
        {"type":"dahai","actor":3,"pai":"2m","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"2p"}
        {"type":"dahai","actor":0,"pai":"9p","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"7p"}
        {"type":"dahai","actor":1,"pai":"1p","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"4p"}
        {"type":"dahai","actor":2,"pai":"W","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"P"}
        {"type":"dahai","actor":3,"pai":"P","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"6m"}
        {"type":"dahai","actor":0,"pai":"W","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"C"}
        {"type":"dahai","actor":1,"pai":"P","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"8m"}
        {"type":"dahai","actor":2,"pai":"9p","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"9m"}
        {"type":"dahai","actor":3,"pai":"9m","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"1p"}
        {"type":"dahai","actor":0,"pai":"2p","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"7m"}
        {"type":"dahai","actor":1,"pai":"S","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"P"}
        {"type":"dahai","actor":2,"pai":"9s","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"N"}
        {"type":"dahai","actor":3,"pai":"N","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"6p"}
        {"type":"dahai","actor":0,"pai":"7p","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"9m"}
        {"type":"dahai","actor":1,"pai":"C","tsumogiri":false}
        {"type":"pon","actor":2,"target":1,"pai":"C","consumed":["C","C"]}
        {"type":"dahai","actor":2,"pai":"1m","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"7s"}
        {"type":"dahai","actor":3,"pai":"7s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"2p"}
        {"type":"dahai","actor":0,"pai":"2p","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"5pr"}
        {"type":"dahai","actor":1,"pai":"9m","tsumogiri":false}
        {"type":"chi","actor":2,"target":1,"pai":"9m","consumed":["7m","8m"]}
        {"type":"dahai","actor":2,"pai":"S","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"E"}
        {"type":"dahai","actor":3,"pai":"E","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"5m"}
        {"type":"dahai","actor":0,"pai":"7s","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"3p"}
        {"type":"dahai","actor":1,"pai":"5p","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"F"}
        {"type":"dahai","actor":2,"pai":"F","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"2s"}
        {"type":"dahai","actor":3,"pai":"2s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"4s"}
        {"type":"dahai","actor":0,"pai":"4s","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"1p"}
        {"type":"dahai","actor":1,"pai":"1p","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"6s"}
        {"type":"dahai","actor":2,"pai":"5m","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"6p"}
        {"type":"dahai","actor":3,"pai":"6p","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"9p"}
        {"type":"dahai","actor":0,"pai":"9p","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"5p"}
        {"type":"dahai","actor":1,"pai":"5p","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"5s"}
        {"type":"dahai","actor":2,"pai":"5s","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"9s"}
        {"type":"dahai","actor":3,"pai":"9s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"8m"}
        {"type":"dahai","actor":0,"pai":"8m","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"9m"}
        {"type":"dahai","actor":1,"pai":"9m","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"9s"}
        {"type":"dahai","actor":2,"pai":"9s","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"1s"}
        {"type":"dahai","actor":3,"pai":"1s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"2m"}
        {"type":"dahai","actor":0,"pai":"5m","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"8m"}
        {"type":"dahai","actor":1,"pai":"8m","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"8p"}
        {"type":"dahai","actor":2,"pai":"8p","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"7m"}
        {"type":"dahai","actor":3,"pai":"7m","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"7p"}
        {"type":"dahai","actor":0,"pai":"7p","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"8p"}
        {"type":"dahai","actor":1,"pai":"7m","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"3m"}
        {"type":"dahai","actor":2,"pai":"3m","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"1s"}
        {"type":"dahai","actor":3,"pai":"1s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"4p"}
        {"type":"dahai","actor":0,"pai":"2m","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"F"}
        {"type":"dahai","actor":1,"pai":"F","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"9s"}
        {"type":"dahai","actor":2,"pai":"9s","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"7m"}
        {"type":"dahai","actor":3,"pai":"7m","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"F"}
        {"type":"dahai","actor":0,"pai":"F","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"8s"}
        {"type":"dahai","actor":1,"pai":"8s","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"F"}
        {"type":"dahai","actor":2,"pai":"F","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"1m"}
        {"type":"dahai","actor":3,"pai":"1m","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"W"}
        {"type":"dahai","actor":0,"pai":"W","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"9m"}
        {"type":"dahai","actor":1,"pai":"9m","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"2m"}
        {"type":"dahai","actor":2,"pai":"2m","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"7p"}
        {"type":"dahai","actor":3,"pai":"7p","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"3p"}
        {"type":"dahai","actor":0,"pai":"6m","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"6m"}
        {"type":"dahai","actor":1,"pai":"6m","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"1s"}
        {"type":"dahai","actor":2,"pai":"1s","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"8m"}
        {"type":"dahai","actor":3,"pai":"8m","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"S"}
        {"type":"dahai","actor":0,"pai":"S","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"2m"}
        {"type":"dahai","actor":1,"pai":"2m","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"4s"}
        {"type":"dahai","actor":2,"pai":"6s","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"8s"}
        {"type":"dahai","actor":3,"pai":"8s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"N"}
        {"type":"dahai","actor":0,"pai":"N","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"3s"}
    "#;
    let ps = PlayerState::from_log(1, log);

    let expected = t![7p, 8p];
    ps.discard_candidates_with_unconditional_tenpai()
        .iter()
        .enumerate()
        .for_each(|(idx, &b)| {
            if expected.contains(&must_tile!(idx)) {
                assert!(b);
            } else {
                assert!(!b);
            }
        });

    // 別のテストケース: 特定の待ち牌でテンパイしている場合
    let log = r#"
        {"type":"start_kyoku","bakaze":"E","dora_marker":"2p","kyoku":4,"honba":0,"kyotaku":0,"oya":3,"scores":[25000,20100,24000,30900],"tehais":[["1m","1m","4m","5m","5m","1p","4p","6p","7p","4s","5s","6s","S"],["5m","6p","7p","2s","3s","4s","4s","5s","7s","9s","S","C","C"],["2m","3m","6m","7m","9m","9m","1p","6p","1s","6s","9s","P","P"],["5mr","6m","8m","8m","2p","5p","7p","8p","9p","3s","9s","W","N"]]}
        {"type":"tsumo","actor":3,"pai":"C"}
        {"type":"dahai","actor":3,"pai":"9s","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"E"}
        {"type":"dahai","actor":0,"pai":"1p","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"2m"}
        {"type":"dahai","actor":1,"pai":"2m","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"9s"}
        {"type":"dahai","actor":2,"pai":"1s","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"8p"}
        {"type":"dahai","actor":3,"pai":"N","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"P"}
        {"type":"dahai","actor":0,"pai":"E","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"3m"}
        {"type":"dahai","actor":1,"pai":"3m","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"8s"}
        {"type":"dahai","actor":2,"pai":"1p","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"S"}
        {"type":"dahai","actor":3,"pai":"S","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"N"}
        {"type":"dahai","actor":0,"pai":"N","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"5pr"}
        {"type":"dahai","actor":1,"pai":"5m","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"1s"}
        {"type":"dahai","actor":2,"pai":"1s","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"9p"}
        {"type":"dahai","actor":3,"pai":"W","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"2p"}
        {"type":"dahai","actor":0,"pai":"P","tsumogiri":false}
        {"type":"pon","actor":2,"target":0,"pai":"P","consumed":["P","P"]}
        {"type":"dahai","actor":2,"pai":"6p","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"3p"}
        {"type":"dahai","actor":3,"pai":"C","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"7m"}
        {"type":"dahai","actor":0,"pai":"S","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"2m"}
        {"type":"dahai","actor":1,"pai":"2m","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"3s"}
        {"type":"dahai","actor":2,"pai":"3s","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"3p"}
        {"type":"dahai","actor":3,"pai":"3s","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"8s"}
        {"type":"dahai","actor":0,"pai":"7m","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"F"}
        {"type":"dahai","actor":1,"pai":"S","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"E"}
        {"type":"dahai","actor":2,"pai":"6s","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"4s"}
        {"type":"dahai","actor":3,"pai":"4s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"7s"}
        {"type":"dahai","actor":0,"pai":"4p","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"6s"}
        {"type":"dahai","actor":1,"pai":"F","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"7m"}
        {"type":"dahai","actor":2,"pai":"8s","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"6m"}
        {"type":"dahai","actor":3,"pai":"2p","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"3p"}
        {"type":"dahai","actor":0,"pai":"1m","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"6p"}
        {"type":"dahai","actor":1,"pai":"6p","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"N"}
        {"type":"dahai","actor":2,"pai":"9s","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"2p"}
        {"type":"dahai","actor":3,"pai":"2p","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"4p"}
        {"type":"dahai","actor":0,"pai":"1m","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"F"}
        {"type":"dahai","actor":1,"pai":"F","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"3m"}
        {"type":"dahai","actor":2,"pai":"9s","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"8p"}
        {"type":"dahai","actor":3,"pai":"5p","tsumogiri":false}
        {"type":"chi","actor":0,"target":3,"pai":"5p","consumed":["6p","7p"]}
        {"type":"dahai","actor":0,"pai":"4m","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"1p"}
        {"type":"dahai","actor":1,"pai":"1p","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"5s"}
        {"type":"dahai","actor":2,"pai":"5s","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"9m"}
        {"type":"dahai","actor":3,"pai":"9m","tsumogiri":true}
        {"type":"pon","actor":2,"target":3,"pai":"9m","consumed":["9m","9m"]}
        {"type":"dahai","actor":2,"pai":"E","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"7s"}
        {"type":"dahai","actor":3,"pai":"7s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"3m"}
        {"type":"dahai","actor":0,"pai":"3m","tsumogiri":true}
        {"type":"pon","actor":2,"target":0,"pai":"3m","consumed":["3m","3m"]}
        {"type":"dahai","actor":2,"pai":"N","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"1s"}
        {"type":"dahai","actor":3,"pai":"1s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"7p"}
        {"type":"dahai","actor":0,"pai":"7p","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"9m"}
        {"type":"dahai","actor":1,"pai":"9m","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"4m"}
        {"type":"dahai","actor":2,"pai":"2m","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"P"}
        {"type":"dahai","actor":3,"pai":"P","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"W"}
        {"type":"dahai","actor":0,"pai":"W","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"F"}
        {"type":"dahai","actor":1,"pai":"F","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"8m"}
        {"type":"dahai","actor":2,"pai":"8m","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"7s"}
        {"type":"dahai","actor":3,"pai":"7s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"4p"}
        {"type":"dahai","actor":0,"pai":"4p","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"3p"}
        {"type":"dahai","actor":1,"pai":"9s","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"8s"}
        {"type":"dahai","actor":2,"pai":"8s","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"2s"}
        {"type":"dahai","actor":3,"pai":"2s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"4p"}
        {"type":"dahai","actor":0,"pai":"4p","tsumogiri":true}
        {"type":"chi","actor":1,"target":0,"pai":"4p","consumed":["3p","5pr"]}
        {"type":"dahai","actor":1,"pai":"7s","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"5p"}
        {"type":"dahai","actor":2,"pai":"5p","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"1m"}
        {"type":"dahai","actor":3,"pai":"8p","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"W"}
        {"type":"dahai","actor":0,"pai":"W","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"8s"}
        {"type":"dahai","actor":1,"pai":"8s","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"8p"}
        {"type":"dahai","actor":2,"pai":"8p","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"F"}
        {"type":"dahai","actor":3,"pai":"1m","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"1p"}
        {"type":"dahai","actor":0,"pai":"1p","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"1m"}
        {"type":"dahai","actor":1,"pai":"1m","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"5sr"}
        {"type":"dahai","actor":2,"pai":"7m","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"9p"}
        {"type":"dahai","actor":3,"pai":"F","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"1s"}
        {"type":"dahai","actor":0,"pai":"1s","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"6s"}
    "#;
    // プレイヤー1の状態をログから再現
    let ps = PlayerState::from_log(1, log);

    // 待ち牌の確認: 5p, 8p待ち
    let expected = t![5p, 8p];
    for (idx, &b) in ps.waits.iter().enumerate() {
        if expected.contains(&must_tile!(idx)) {
            assert!(b); // 5pと8pは待ち牌
        } else {
            assert!(!b); // それ以外は待ち牌ではない
        }
    }

    // 特定の待ち牌の場合、無条件テンパイの捨て牌候補はない
    let discard_candidates = ps.discard_candidates_with_unconditional_tenpai();
    assert_eq!(discard_candidates, [false; 34]); // 全ての牌がfalse（捨てられない）
}

// 搶槓（チャンカン）ロンのテスト
// 他家が加槓した牌でロンできるか検証
#[test]
fn double_chankan_ron() {
    // 南二局のゲームログ
    let log = r#"
        {"type":"start_kyoku","bakaze":"S","dora_marker":"2p","kyoku":2,"honba":0,"kyotaku":0,"oya":1,"scores":[44400,1600,25700,28300],"tehais":[["1m","5m","9m","9m","9m","3p","9p","8s","9s","W","W","N","C"],["7m","8m","3p","6p","8p","1s","1s","3s","6s","9s","E","F","C"],["3m","9m","2p","5p","8p","1s","2s","5s","6s","7s","S","F","C"],["2m","2m","5m","5mr","8m","1p","1p","7p","8p","3s","5s","8s","9s"]]}
        {"type":"tsumo","actor":1,"pai":"P"}
        {"type":"dahai","actor":1,"pai":"F","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"3m"}
        {"type":"dahai","actor":2,"pai":"F","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"6m"}
        {"type":"dahai","actor":3,"pai":"9s","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"1s"}
        {"type":"dahai","actor":0,"pai":"1s","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"9p"}
        {"type":"dahai","actor":1,"pai":"C","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"9p"}
        {"type":"dahai","actor":2,"pai":"C","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"7s"}
        {"type":"dahai","actor":3,"pai":"1p","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"7p"}
        {"type":"dahai","actor":0,"pai":"C","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"5m"}
        {"type":"dahai","actor":1,"pai":"P","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"8s"}
        {"type":"dahai","actor":2,"pai":"9m","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"7m"}
        {"type":"dahai","actor":3,"pai":"1p","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"W"}
        {"type":"dahai","actor":0,"pai":"1m","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"P"}
        {"type":"dahai","actor":1,"pai":"P","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"4m"}
        {"type":"dahai","actor":2,"pai":"S","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"8m"}
        {"type":"dahai","actor":3,"pai":"8m","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"8p"}
        {"type":"dahai","actor":0,"pai":"N","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"5sr"}
        {"type":"dahai","actor":1,"pai":"E","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"E"}
        {"type":"dahai","actor":2,"pai":"E","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"4p"}
        {"type":"dahai","actor":3,"pai":"4p","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"1m"}
        {"type":"dahai","actor":0,"pai":"5m","tsumogiri":false}
        {"type":"pon","actor":3,"target":0,"pai":"5m","consumed":["5m","5mr"]}
        {"type":"dahai","actor":3,"pai":"8s","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"4s"}
        {"type":"dahai","actor":0,"pai":"4s","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"N"}
        {"type":"dahai","actor":1,"pai":"N","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"9p"}
        {"type":"dahai","actor":2,"pai":"8p","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"C"}
        {"type":"dahai","actor":3,"pai":"C","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"4s"}
        {"type":"dahai","actor":0,"pai":"4s","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"1m"}
        {"type":"dahai","actor":1,"pai":"9s","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"4p"}
        {"type":"dahai","actor":2,"pai":"2p","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"P"}
        {"type":"dahai","actor":3,"pai":"P","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"3m"}
        {"type":"dahai","actor":0,"pai":"3p","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"6s"}
        {"type":"dahai","actor":1,"pai":"9p","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"8s"}
        {"type":"dahai","actor":2,"pai":"3m","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"4m"}
        {"type":"dahai","actor":3,"pai":"4m","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"P"}
        {"type":"dahai","actor":0,"pai":"P","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"E"}
        {"type":"dahai","actor":1,"pai":"E","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"7s"}
        {"type":"dahai","actor":2,"pai":"2s","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"F"}
        {"type":"dahai","actor":3,"pai":"F","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"4m"}
        {"type":"dahai","actor":0,"pai":"4m","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"2m"}
        {"type":"dahai","actor":1,"pai":"5m","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"7p"}
        {"type":"dahai","actor":2,"pai":"7p","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"2s"}
        {"type":"dahai","actor":3,"pai":"2s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"4p"}
        {"type":"dahai","actor":0,"pai":"4p","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"5pr"}
        {"type":"dahai","actor":1,"pai":"8p","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"2s"}
        {"type":"dahai","actor":2,"pai":"2s","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"F"}
        {"type":"dahai","actor":3,"pai":"F","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"6p"}
        {"type":"dahai","actor":0,"pai":"6p","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"7m"}
        {"type":"dahai","actor":1,"pai":"3p","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"1p"}
        {"type":"dahai","actor":2,"pai":"1p","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"9s"}
        {"type":"dahai","actor":3,"pai":"9s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"S"}
        {"type":"dahai","actor":0,"pai":"S","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"7s"}
        {"type":"dahai","actor":1,"pai":"6s","tsumogiri":false}
        {"type":"chi","actor":2,"target":1,"pai":"6s","consumed":["5s","7s"]}
        {"type":"dahai","actor":2,"pai":"1s","tsumogiri":false}
        {"type":"pon","actor":1,"target":2,"pai":"1s","consumed":["1s","1s"]}
        {"type":"dahai","actor":1,"pai":"3s","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"2p"}
        {"type":"dahai","actor":2,"pai":"2p","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"3p"}
        {"type":"dahai","actor":3,"pai":"3p","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"6s"}
        {"type":"dahai","actor":0,"pai":"6s","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"6p"}
        {"type":"dahai","actor":1,"pai":"6p","tsumogiri":true}
        {"type":"chi","actor":2,"target":1,"pai":"6p","consumed":["4p","5p"]}
        {"type":"dahai","actor":2,"pai":"8s","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"6m"}
        {"type":"dahai","actor":3,"pai":"3s","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"7m"}
        {"type":"dahai","actor":0,"pai":"8s","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"6p"}
        {"type":"dahai","actor":1,"pai":"6p","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"5s"}
        {"type":"dahai","actor":2,"pai":"8s","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"1p"}
        {"type":"dahai","actor":3,"pai":"1p","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"2s"}
        {"type":"dahai","actor":0,"pai":"9s","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"1m"}
        {"type":"dahai","actor":1,"pai":"2m","tsumogiri":false}
        {"type":"pon","actor":3,"target":1,"pai":"2m","consumed":["2m","2m"]}
        {"type":"dahai","actor":3,"pai":"6m","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"W"}
        {"type":"dahai","actor":0,"pai":"2s","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"N"}
        {"type":"dahai","actor":1,"pai":"N","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"5p"}
        {"type":"dahai","actor":2,"pai":"5p","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"3m"}
        {"type":"dahai","actor":3,"pai":"3m","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"6m"}
        {"type":"ankan","actor":0,"consumed":["W","W","W","W"]}
        {"type":"dora","dora_marker":"7p"}
        {"type":"tsumo","actor":0,"pai":"8m"}
        {"type":"dahai","actor":0,"pai":"6m","tsumogiri":false}
        {"type":"chi","actor":1,"target":0,"pai":"6m","consumed":["7m","8m"]}
        {"type":"dahai","actor":1,"pai":"7m","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"3s"}
        {"type":"dahai","actor":2,"pai":"3s","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"2m"}
    "#;
    // プレイヤー2の状態をログから再現
    let mut ps = PlayerState::from_log(2, log);

    // プレイヤー3が2mを加槓した場合のテスト
    let mut ps_kakan = ps.clone();
    let cans = ps_kakan
        .test_update_json(r#"{"type":"kakan","actor":3,"pai":"2m","consumed":["2m","2m","2m"]}"#);
    assert!(cans.can_ron_agari); // 搶槓ロン可能
    assert_eq!(ps_kakan.agari_points(true, &[]).unwrap().ron, 1000); // ロン点数: 1000点

    // 通常の打牌の場合はロンできない
    let cans = ps.test_update_json(r#"{"type":"dahai","actor":3,"pai":"2m","tsumogiri":true}"#);
    assert!(!cans.can_ron_agari); // 通常の打牌ではロン不可
}

// 0シャンテン時のチー判定のテスト
// テンパイ時にチーして和了する場合の処理を検証
#[test]
fn chi_at_0_shanten() {
    // 東一局のゲームログ
    let log = r#"
        {"type":"start_kyoku","bakaze":"E","dora_marker":"W","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"tehais":[["1m","2m","3m","5p","5p","4s","5s","E","E","E","S","S","S"],["?","?","?","?","?","?","?","?","?","?","?","?","?"],["?","?","?","?","?","?","?","?","?","?","?","?","?"],["?","?","?","?","?","?","?","?","?","?","?","?","?"]]}
        {"type":"tsumo","actor":0,"pai":"P"}
        {"type":"dahai","actor":0,"pai":"P","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"?"}
        {"type":"dahai","actor":1,"pai":"P","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"?"}
        {"type":"dahai","actor":2,"pai":"P","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"?"}
        {"type":"dahai","actor":3,"pai":"6s","tsumogiri":false}
    "#;
    // プレイヤー0の状態をログから再現
    let mut ps = PlayerState::from_log(0, log);

    // 現在で0シャンテン（テンパイ）
    assert_eq!(ps.shanten, 0);
    assert_eq!(ps.real_time_shanten(), 0);
    assert!(ps.last_cans.can_ron_agari); // 6sでロン可能
    assert!(ps.last_cans.can_chi_high); // 456の上チー可能

    // 6sをチーして和了
    ps.test_update_json(r#"{"type":"chi","actor":0,"target":3,"consumed":["4s","5s"],"pai":"6s"}"#);
    assert_eq!(ps.shanten, 0); // シャンテン数は0のまま
    assert_eq!(ps.real_time_shanten(), -1); // 実際は和了形（-1）
    assert!(ps.at_furiten); // チーしたことでフリテンになる
    assert!(!ps.has_next_shanten_discard); // 次のシャンテンへの捨て牌はない
}
