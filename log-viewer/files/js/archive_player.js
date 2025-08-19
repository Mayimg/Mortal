/**
 * 麻雀牌譜プレイヤー - アーカイブ再生エンジン
 * CoffeeScript 1.6.3から生成されたJavaScriptコード
 * MJAI形式の麻雀牌譜データを読み込み、視覚的に再生する機能を提供
 */

// CoffeeScriptから生成されたコード（変数宣言）
// 主要な関数とグローバル変数をまとめて宣言
var BAKAZE_TO_STR, TSUPAIS, TSUPAI_TO_IMAGE_NAME, cloneBoard, comparePais, currentActionId, currentKyokuId, currentViewpoint, deleteTehai, dumpBoard, getCurrentKyoku, goBack, goNext, initPlayers, kyokus, loadAction, paiToImageUrl, parsePai, playerInfos, removeRed, renderAction, renderCurrentAction, renderHo, renderPai, renderPais, ripai, sortPais, _base, _base1;

// ブラウザ互換性のためのconsoleオブジェクト初期化
// 古いブラウザでconsoleが未定義の場合の対策
window.console || (window.console = {});

// console.logメソッドが存在しない場合のフォールバック
(_base = window.console).log || (_base.log = function() {});

// console.errorメソッドが存在しない場合のフォールバック
(_base1 = window.console).error || (_base1.error = function() {});

/**
 * 字牌（じはい）の定義配列
 * インデックス0はnull、1から順に東南西北白発中に対応
 * MJAI形式の牌表現と配列インデックスの対応表
 */
TSUPAIS = [null, "E", "S", "W", "N", "P", "F", "C"];

/**
 * 字牌のMJAI形式から画像ファイル名への変換マップ
 * 古い画像システムで使用されていた形式（現在は使用されていない）
 */
TSUPAI_TO_IMAGE_NAME = {
  "E": "ji_e",    // 東
  "S": "ji_s",    // 南
  "W": "ji_w",    // 西
  "N": "ji_n",    // 北
  "P": "no",      // 白
  "F": "ji_h",    // 発
  "C": "ji_c"     // 中
};

/**
 * 場風（ばかぜ）のMJAI形式から日本語表記への変換マップ
 * ゲームのUIで「東1局」「南2局」などの表示に使用
 */
BAKAZE_TO_STR = {
  "E": "東",
  "S": "南",
  "W": "西",
  "N": "北"
};

/**
 * ゲーム状態を管理するグローバル変数群
 */

// 全局（きょく）のデータを格納する配列
// 各局には開始から終了までの全アクションが含まれる
kyokus = [];

// 現在表示中の局のID（0から始まるインデックス）
currentKyokuId = 0;

// 現在表示中のアクションのID（局内での0から始まるインデックス）
currentActionId = 0;

// 現在の視点プレイヤー（0:下座, 1:右座, 2:上座, 3:左座）
// 画面下部に表示されるプレイヤーを決定する
currentViewpoint = 0;

// 4人のプレイヤー情報を格納する配列
// 各要素はプレイヤーの名前、スコアなどの情報を持つオブジェクト
playerInfos = [{}, {}, {}, {}];

/**
 * 牌（ぱい）の文字列表現を解析して構造化データに変換する関数
 * MJAI形式の牌表現（例："1m", "5sr", "E"）を解析
 * @param {string} pai - 解析する牌の文字列（例："1m", "5pr", "E"）
 * @returns {Object} 解析結果 {type: 牌の種類, number: 数字, red: 赤牌フラグ}
 */
parsePai = function(pai) {
  // 数牌（1-9 + 種類 + 赤牌フラグ）の正規表現チェック
  // 例："1m"（一萬）、"5pr"（赤五筒）
  if (pai.match(/^([1-9])(.)(r)?$/)) {
    return {
      type: RegExp.$2,                    // 牌の種類（m:萬子, p:筒子, s:索子）
      number: parseInt(RegExp.$1),        // 牌の数字（1-9）
      red: RegExp.$3 ? true : false       // 赤牌かどうか（"r"が付いているか）
    };
  } else {
    // 字牌（東南西北白発中）の場合
    return {
      type: "t",                         // 字牌を表す"t"（Terminal & honor）
      number: TSUPAIS.indexOf(pai),       // TSUPAIS配列でのインデックス
      red: false                          // 字牌に赤牌は存在しない
    };
  }
};

/**
 * 2つの牌を比較してソート順序を決定する関数
 * 手牌の表示順序を正しくするために使用
 * @param {string} lhs - 比較する牌1
 * @param {string} rhs - 比較する牌2
 * @returns {number} -1（lhs < rhs）, 0（lhs = rhs）, 1（lhs > rhs）
 */
comparePais = function(lhs, rhs) {
  var lhsRep, parsedLhs, parsedRhs, rhsRep;
  
  // 両方の牌を解析
  parsedLhs = parsePai(lhs);
  parsedRhs = parsePai(rhs);
  
  // 比較用の文字列を生成（種類 + 数字 + 赤牌フラグ）
  // 例："m31"（赤三萬）、"t10"（東）
  lhsRep = parsedLhs.type + parsedLhs.number + (parsedLhs.red ? "1" : "0");
  rhsRep = parsedRhs.type + parsedRhs.number + (parsedRhs.red ? "1" : "0");
  
  // 文字列の辞書順比較でソート順序を決定
  if (lhsRep < rhsRep) {
    return -1;  // lhsの方が小さい
  } else if (lhsRep > rhsRep) {
    return 1;   // rhsの方が小さい
  } else {
    return 0;   // 同じ
  }
};

/**
 * 牌の配列を正しい順序でソートする関数
 * 手牌を萬子→筒子→索子→字牌の順に並べる
 * @param {Array<string>} pais - ソートする牌の配列
 * @returns {Array<string>} ソート済みの牌配列
 */
sortPais = function(pais) {
  return pais.sort(comparePais);
};

/**
 * 牌のMJAI表現から対応する画像ファイルのURLを生成する関数
 * 新しい画像システム（files/Export/New_Regular/）を使用
 * @param {string|null} pai - 牌の文字列表現または「?」（裏向き）、null
 * @param {number} pose - 牌のポーズ（現在のシステムでは無視される）
 * @returns {string|null} 画像ファイルのURL、またはnull
 */
paiToImageUrl = function(pai, pose) {
  var name, parsedPai;
  
  if (pai) {
    if (pai === "?") {
      // 裏向きの牌（他のプレイヤーの手牌など）
      name = "Back";
    } else {
      parsedPai = parsePai(pai);
      
      if (parsedPai.type === "t") {
        // 字牌（東南西北白発中）の画像ファイル名を決定
        switch(pai) {
          case "E": name = "Ton"; break;    // 東
          case "S": name = "Nan"; break;    // 南
          case "W": name = "Shaa"; break;   // 西
          case "N": name = "Pei"; break;    // 北
          case "P": name = "Haku"; break;   // 白
          case "F": name = "Hatsu"; break;  // 発
          case "C": name = "Chun"; break;   // 中
        }
      } else {
        // 数牌（萬子・筒子・索子）の画像ファイル名を決定
        var typePrefix;
        switch(parsedPai.type) {
          case "m": typePrefix = "Man"; break;  // 萬子
          case "p": typePrefix = "Pin"; break;  // 筒子
          case "s": typePrefix = "Sou"; break;  // 索子
        }
        
        if (parsedPai.red) {
          // 赤牌の場合は"-Dora"サフィックスを追加
          name = typePrefix + parsedPai.number + "-Dora";
        } else {
          // 通常の牌
          name = typePrefix + parsedPai.number;
        }
      }
    }
    
    // 注意：poseパラメータは新システムでは無視される - 全ての画像が.png形式
    return "files/Export/New_Regular/" + name + ".png";
  } else {
    // null値の場合はnullを返す（Blank.pngは返さない）
    return null;
  }
};

/**
 * 盤面（ゲーム状態）オブジェクトの深いコピーを作成する関数
 * プレイヤー情報は個別にクローンされ、それ以外は浅いコピー
 * ゲーム状態の履歴管理で各アクション時点の状態を保存するために使用
 * @param {Object} board - クローンする盤面オブジェクト
 * @returns {Object} クローンされた盤面オブジェクト
 */
cloneBoard = function(board) {
  var bk, bv, newBoard, newPlayer, pk, player, pv, _i, _len;
  
  newBoard = {};
  
  // 盤面オブジェクトの各プロパティを処理
  for (bk in board) {
    bv = board[bk];
    
    if (bk === "players") {
      // プレイヤー配列は特別扱い：各プレイヤーオブジェクトを個別にクローン
      newBoard[bk] = [];
      for (_i = 0, _len = bv.length; _i < _len; _i++) {
        player = bv[_i];
        newPlayer = {};
        
        // プレイヤーオブジェクトの各プロパティをコピー
        for (pk in player) {
          pv = player[pk];
          newPlayer[pk] = pv;
        }
        
        newBoard[bk].push(newPlayer);
      }
    } else {
      // プレイヤー以外のプロパティは浅いコピー
      newBoard[bk] = bv;
    }
  }
  
  return newBoard;
};

/**
 * プレイヤーオブジェクトを初期状態に設定する関数
 * 新しい局が開始される際に各プレイヤーの状態をリセット
 * @param {Object} board - 初期化する盤面オブジェクト
 * @returns {Array} 処理結果の配列
 */
initPlayers = function(board) {
  var player, _i, _len, _ref, _results;
  
  _ref = board.players;
  _results = [];
  
  // 各プレイヤー（4人）の状態を初期化
  for (_i = 0, _len = _ref.length; _i < _len; _i++) {
    player = _ref[_i];
    
    player.tehais = null;           // 手牌：後でstart_kyokuアクションで設定される
    player.furos = [];              // 副露（ポン・チー・カン）：空配列で初期化
    player.ho = [];                 // 河（捨て牌）：空配列で初期化
    player.reach = false;           // リーチ状態：初期はfalse
    _results.push(player.reachHoIndex = null); // リーチ時の河インデックス：初期はnull
  }
  
  return _results;
};

/**
 * 牌の文字列表現から赤牌マーカー（"r"）を除去する関数
 * 赤牌と通常牌を同一視したい場合（カカンの判定など）に使用
 * @param {string|null} pai - 処理する牌の文字列
 * @returns {string|null} 赤牌マーカーを除去した牌文字列、またはnull
 */
removeRed = function(pai) {
  if (!pai) {
    return null;
  }
  
  // 末尾に"r"が付いている場合（赤牌）はそれを除去
  if (pai.match(/^(.+)r$/)) {
    return RegExp.$1;  // "5pr" → "5p"
  } else {
    return pai;        // 通常牌はそのまま
  }
};

/**
 * MJAIアクションを読み込んで内部データ構造に変換する関数
 * 各アクションタイプに応じて適切な盤面更新処理を実行
 * @param {Object} action - 処理するMJAIアクション
 * @returns {void}
 */
loadAction = function(action) {
  var actorPlayer, board, furos, i, kyoku, pai, prevBoard, targetPlayer, _i, _j, _k, _l, _len, _len1, _m, _n, _o, _ref, _ref1, _ref2;
  
  // 現在の局と盤面状態を取得
  if (kyokus.length > 0) {
    kyoku = kyokus[kyokus.length - 1];  // 最新の局
    // 最新アクションの盤面状態をクローンして新しいベースとする
    board = cloneBoard(kyoku.actions[kyoku.actions.length - 1].board);
  } else {
    // まだ局が開始されていない場合
    kyoku = null;
    board = null;
  }
  
  // アクション実行プレイヤーの取得
  if (board && ("actor" in action)) {
    actorPlayer = board.players[action.actor];
  } else {
    actorPlayer = null;
  }
  
  // アクション対象プレイヤーの取得（副露の相手など）
  if (board && ("target" in action)) {
    targetPlayer = board.players[action.target];
  } else {
    targetPlayer = null;
  }
  // アクションタイプに応じた処理の分岐
  switch (action.type) {
    case "start_game":
      // ゲーム開始：プレイヤー名を設定
      for (i = _i = 0; _i < 4; i = ++_i) {
        playerInfos[i].name = action.names[i];
      }
      break;
      
    case "end_game":
      // ゲーム終了：特別な処理なし
      null;
      break;
      
    case "start_kyoku":
      // 局開始：新しい局オブジェクトを作成
      kyoku = {
        actions: [],                    // この局のアクション履歴
        bakaze: action.bakaze,          // 場風（E/S/W/N）
        kyokuNum: action.kyoku,         // 局数（1-4）
        honba: action.honba             // 本場数
      };
      kyokus.push(kyoku);
      
      prevBoard = board;  // 前局の盤面を保存
      
      // 新しい盤面を初期化
      board = {
        players: [{}, {}, {}, {}],               // 4人のプレイヤー
        doraMarkers: [action.dora_marker]        // ドラ表示牌
      };
      
      initPlayers(board);  // プレイヤー状態を初期化
      
      // 各プレイヤーの配牌とスコアを設定
      for (i = _j = 0; _j < 4; i = ++_j) {
        board.players[i].tehais = action.tehais[i];  // 配牌
        
        if (prevBoard) {
          // 前局のスコアを引き継ぎ
          board.players[i].score = prevBoard.players[i].score;
        } else {
          // 初回は25000点からスタート
          board.players[i].score = 25000;
        }
      }
      break;
    case "end_kyoku":
      // 局終了：特別な処理なし
      null;
      break;
      
    case "tsumo":
      // ツモ：プレイヤーの手牌に牌を追加
      actorPlayer.tehais = actorPlayer.tehais.concat([action.pai]);
      break;
      
    case "dahai":
      // 打牌：手牌から牌を削除し、河に追加
      deleteTehai(actorPlayer, action.pai);
      actorPlayer.ho = actorPlayer.ho.concat([action.pai]);
      break;
      
    case "reach":
      // リーチ宣言：次に捨てる牌の位置を記録
      actorPlayer.reachHoIndex = actorPlayer.ho.length;
      break;
      
    case "reach_accepted":
      // リーチ受理：プレイヤーのリーチ状態をtrueに
      actorPlayer.reach = true;
      break;
    case "chi":
    case "pon":
    case "daiminkan":
      // チー・ポン・大明カン：他プレイヤーの最後の捨て牌を取得
      
      // 対象プレイヤーの河から最後の牌を除去
      targetPlayer.ho = targetPlayer.ho.slice(0, targetPlayer.ho.length - 1);
      
      // アクションプレイヤーの手牌から消費牌を削除
      _ref = action.consumed;
      for (_k = 0, _len = _ref.length; _k < _len; _k++) {
        pai = _ref[_k];
        deleteTehai(actorPlayer, pai);
      }
      
      // 副露情報を追加
      actorPlayer.furos = actorPlayer.furos.concat([
        {
          type: action.type,          // 副露の種類（chi/pon/daiminkan）
          taken: action.pai,          // 取得した牌
          consumed: action.consumed,  // 手牌から消費した牌
          target: action.target       // 牌を取得した相手プレイヤー
        }
      ]);
      break;
    case "ankan":
      // 暗カン：手牌から4枚の同じ牌を取り除く
      
      _ref1 = action.consumed;
      for (_l = 0, _len1 = _ref1.length; _l < _len1; _l++) {
        pai = _ref1[_l];
        deleteTehai(actorPlayer, pai);
      }
      
      // 暗カンの副露情報を追加（takenとtargetは不要）
      actorPlayer.furos = actorPlayer.furos.concat([
        {
          type: action.type,          // "ankan"
          consumed: action.consumed   // 消費した4枚の牌
        }
      ]);
      break;
    case "kakan":
      // 加カン：既存のポンに1枚追加してカンに変更
      
      // 手牌から加カン牌を削除
      deleteTehai(actorPlayer, action.pai);
      
      // 空の配列を結合（CoffeeScript由来の不要な処理）
      actorPlayer.furos = actorPlayer.furos.concat([]);
      
      furos = actorPlayer.furos;
      
      // 既存のポンを見つけて加カンに変更
      for (i = _m = 0, _ref2 = furos.length; 0 <= _ref2 ? _m < _ref2 : _m > _ref2; i = 0 <= _ref2 ? ++_m : --_m) {
        // 赤牌を無視して牌の種類が一致するポンを検索
        if (furos[i].type === "pon" && removeRed(furos[i].taken) === removeRed(action.pai)) {
          // ポンを加カンに置き換え
          furos[i] = {
            type: "kakan",              // 加カン
            taken: action.pai,          // 加カンした牌
            consumed: action.consumed,  // 元のポン牌 + 加カン牌
            target: furos[i].target     // 元のポンの相手プレイヤー
          };
        }
      }
      break;
    case "hora":
    case "ryukyoku":
      // 和了・流局：特別な処理なし（スコア更新は後で行われる）
      null;
      break;
      
    case "dora":
      // ドラ追加：新しいドラ表示牌を追加
      board.doraMarkers = board.doraMarkers.concat([action.dora_marker]);
      break;
      
    case "error":
      // エラー：特別な処理なし
      null;
      break;
      
    default:
      // 未知のアクションタイプ
      throw "unknown action: " + action.type;
  }
  // スコア更新処理（アクションにscores情報がある場合）
  if (action.scores) {
    for (i = _n = 0; _n < 4; i = ++_n) {
      board.players[i].score = action.scores[i];
    }
  }
  
  // 局が存在する場合の後処理
  if (kyoku) {
    // アクション実行者以外のプレイヤーの手牌を整理（理牌）
    for (i = _o = 0; _o < 4; i = ++_o) {
      if (action.actor !== void 0 && i !== action.actor) {
        ripai(board.players[i]);
      }
    }
    
    // アクションに盤面状態を保存
    action.board = board;
    
    // 局のアクション履歴に追加
    return kyoku.actions.push(action);
  }
};

/**
 * プレイヤーの手牌から指定された牌を1枚削除する関数
 * 打牌や副露時に手牌から牌を取り除く際に使用
 * @param {Object} player - 対象プレイヤーオブジェクト
 * @param {string} pai - 削除する牌の文字列
 * @returns {null} 削除された位置にnullを設定
 */
deleteTehai = function(player, pai) {
  var idx;
  
  // 手牌配列の浅いコピーを作成（元配列を変更しないため）
  player.tehais = player.tehais.concat([]);
  
  // 指定された牌を手牌の後ろから検索（後入れ先出し）
  idx = player.tehais.lastIndexOf(pai);
  
  if (idx < 0) {
    // 指定された牌が見つからない場合、裏向き牌（"?"）を検索
    // 他のプレイヤーの視点では手牌が"?"で表現されることがある
    idx = player.tehais.lastIndexOf("?");
  }
  
  if (idx < 0) {
    // どちらも見つからない場合はエラー
    throw "pai not in tehai";
  }
  
  // 見つかった位置の牌をnullに設定（削除マーク）
  return player.tehais[idx] = null;
};

/**
 * プレイヤーの手牌を整理する関数（理牌 - りーぱい）
 * null要素（削除マーク）を除去し、残った牌を正しい順序でソート
 * @param {Object} player - 整理対象のプレイヤーオブジェクト
 * @returns {Array|undefined} ソート済みの手牌配列、または手牌がない場合はundefined
 */
ripai = function(player) {
  var pai;
  
  if (player.tehais) {
    // null以外の牌のみを抽出する即座実行関数
    player.tehais = (function() {
      var _i, _len, _ref, _results;
      _ref = player.tehais;
      _results = [];
      
      // 各牌をチェックしてnull以外のものだけを新配列に追加
      for (_i = 0, _len = _ref.length; _i < _len; _i++) {
        pai = _ref[_i];
        if (pai) {  // nullやundefinedでない場合
          _results.push(pai);
        }
      }
      
      return _results;
    })();
    
    // 抽出された牌を正しい順序でソート
    return sortPais(player.tehais);
  }
};

/**
 * 盤面の現在状態をコンソールに出力するデバッグ用関数
 * 各プレイヤーの手牌、副露、河の状態を見やすく表示
 * @param {Object} board - 出力する盤面オブジェクト
 * @returns {Array} コンソール出力結果の配列
 */
dumpBoard = function(board) {
  var consumedStr, furo, hoStr, i, player, tehaisStr, _i, _j, _len, _ref, _results;
  
  _results = [];
  
  // 各プレイヤー（4人）の状態を出力
  for (i = _i = 0; _i < 4; i = ++_i) {
    player = board.players[i];
    
    // 手牌と副露の情報を出力
    if (player.tehais) {
      // 手牌を空白区切りの文字列に変換
      tehaisStr = player.tehais.join(" ");
      
      // 各副露を [取得牌/消費牌] 形式で追加
      _ref = player.furos;
      for (_j = 0, _len = _ref.length; _j < _len; _j++) {
        furo = _ref[_j];
        consumedStr = furo.consumed.join(" ");
        tehaisStr += " [" + furo.taken + "/" + consumedStr + "]";
      }
      
      console.log("[" + i + "] tehais: " + tehaisStr);
    }
    
    // 河（捨て牌）の情報を出力
    if (player.ho) {
      hoStr = player.ho.join(" ");
      _results.push(console.log("[" + i + "] ho: " + hoStr));
    } else {
      _results.push(void 0);
    }
  }
  
  return _results;
};

/**
 * 単一の牌を指定されたDOM要素に描画する関数
 * 牌の画像とCSSクラスを適切に設定
 * @param {string|null} pai - 描画する牌の文字列
 * @param {jQuery} view - 描画先のjQuery要素（img要素）
 * @param {number} pose - 牌のポーズ（1:縦向き, 3:横向き）
 * @returns {void}
 */
renderPai = function(pai, view, pose) {
  // poseが未指定の場合はデフォルトで縦向き（1）
  if (pose === void 0) {
    pose = 1;
  }
  
  // 牌に対応する画像URLを取得
  var imageUrl = paiToImageUrl(pai, pose);
  
  if (imageUrl === null) {
    // 画像がない場合（null牌など）は要素を非表示
    view.hide();
    return;
  } else {
    // 画像がある場合は要素を表示し、src属性を設定
    view.show();
    view.attr("src", imageUrl);
  }
  
  // ポーズに応じてCSSクラスを設定
  switch (pose) {
    case 1:
      // 縦向き（通常の手牌）
      view.addClass("pai");
      return view.removeClass("laid-pai");
    case 3:
      // 横向き（捨て牌、リーチ牌など）
      view.addClass("laid-pai");
      return view.removeClass("pai");
    default:
      throw "unknown pose";
  }
};

/**
 * 複数の牌を配列として描画する関数
 * Repeatedオブジェクトを使用して動的に要素数を調整
 * @param {Array<string>} pais - 描画する牌の配列
 * @param {Repeated} view - 描画先のRepeatedオブジェクト
 * @param {Array<number>} poses - 各牌のポーズ配列
 * @returns {Array} 各牌の描画結果
 */
renderPais = function(pais, view, poses) {
  var i, _i, _ref, _results;
  
  // 未定義の場合は空配列で初期化
  pais || (pais = []);
  poses || (poses = []);
  
  // Repeatedオブジェクトのサイズを牌の数に合わせて調整
  view.resize(pais.length);
  
  _results = [];
  
  // 各牌を順次描画
  for (i = _i = 0, _ref = pais.length; 0 <= _ref ? _i < _ref : _i > _ref; i = 0 <= _ref ? ++_i : --_i) {
    _results.push(renderPai(pais[i], view.at(i), poses[i]));
  }
  
  return _results;
};

/**
 * プレイヤーの河（捨て牌エリア）を描画する関数
 * リーチ牌は横向きで表示される特別処理を含む
 * @param {Object} player - 対象プレイヤーオブジェクト
 * @param {number} offset - 河の行におけるオフセット（0, 6, 12など）
 * @param {Array<string>} pais - 描画する捨て牌の配列
 * @param {Repeated} view - 描画先のRepeatedオブジェクト
 * @returns {Array} 各牌の描画結果
 */
renderHo = function(player, offset, pais, view) {
  var i, reachIndex, _i, _ref, _results;
  
  // リーチ牌のインデックスを計算
  if (player.reachHoIndex === null) {
    reachIndex = null;  // リーチしていない場合
  } else {
    // 現在の行でのリーチ牌の相対位置を計算
    reachIndex = player.reachHoIndex - offset;
  }
  
  // 表示要素数を牌の数に合わせて調整
  view.resize(pais.length);
  
  _results = [];
  
  // 各捨て牌を描画
  for (i = _i = 0, _ref = pais.length; 0 <= _ref ? _i < _ref : _i > _ref; i = 0 <= _ref ? ++_i : --_i) {
    // リーチ牌（reachIndex位置）は横向き（pose=3）、それ以外は縦向き（pose=1）
    _results.push(renderPai(pais[i], view.at(i), i === reachIndex ? 3 : 1));
  }
  
  return _results;
};

/**
 * 指定されたアクションの状態を画面に描画する関数
 * 盤面、プレイヤー情報、アクション詳細を全て更新
 * @param {Object} action - 描画するアクション（board情報を含む）
 * @returns {void}
 */
renderAction = function(action) {
  var dir, displayAction, furo, furoView, ho, i, infoView, j, k, kyoku, laidPos, pais, player, poses, v, view, wanpais, _i, _j, _ref, _ref1, _ref2, _ref3;
  
  // アクション情報の表示用オブジェクトを作成（boardとlogsを除外）
  displayAction = {};
  for (k in action) {
    v = action[k];
    if (k !== "board" && k !== "logs") {
      displayAction[k] = v;
    }
  }
  
  // アクション詳細をJSON形式で表示
  $("#action-label").text(JSON.stringify(displayAction, null, 2));
  
  // 現在の視点プレイヤーのログメッセージを表示
  $("#log-label").text((action.logs && action.logs[currentViewpoint]) || "");
  kyoku = getCurrentKyoku();
  
  // 各プレイヤー（4人）の状態を描画
  for (i = _i = 0; _i < 4; i = ++_i) {
    player = action.board.players[i];
    
    // 視点に応じてプレイヤーの表示位置を調整
    // 現在の視点プレイヤーが常に下座（position 0）に表示される
    view = Dytem.players.at((i - currentViewpoint + 4) % 4);
    
    // プレイヤー情報テーブルの更新
    infoView = Dytem.playerInfos.at(i);
    infoView.score.text(player.score);  // スコア表示
    infoView.viewpoint.text(i === currentViewpoint ? "+" : "");  // 現在の視点に「+」マーク
    // 手牌の描画処理
    if (!player.tehais) {
      // 手牌が設定されていない場合（ゲーム開始前など）
      renderPais([], view.tehais);
      view.tsumoPai.hide();
    } else if (player.tehais.length % 3 === 2) {
      // 手牌数が3n+2枚の場合：最後の1枚をツモ牌として分離表示
      // 通常時13枚 + ツモ牌1枚 = 14枚（14 % 3 = 2）
      renderPais(player.tehais.slice(0, player.tehais.length - 1), view.tehais);
      view.tsumoPai.show();
      renderPai(player.tehais[player.tehais.length - 1], view.tsumoPai);
    } else {
      // 手牌数が3n枚の場合：全て通常手牌として表示
      // 通常時13枚（13 % 3 = 1だが、実際は13枚で運用）
      renderPais(player.tehais, view.tehais);
      view.tsumoPai.hide();
    }
    // 河（捨て牌）の描画：3行に分けて表示
    ho = player.ho || [];
    renderHo(player, 0, ho.slice(0, 6), view.hoRows.at(0).pais);     // 1行目：0-5番目
    renderHo(player, 6, ho.slice(6, 12), view.hoRows.at(1).pais);    // 2行目：6-11番目
    renderHo(player, 12, ho.slice(12), view.hoRows.at(2).pais);      // 3行目：12番目以降
    // 副露（ポン・チー・カン）の描画
    view.furos.resize(player.furos.length);
    
    if (player.furos) {
      // 副露は逆順で表示（新しい副露が右側に来るように）
      j = player.furos.length - 1;
      while (j >= 0) {
        furo = player.furos[j];
        furoView = view.furos.at(player.furos.length - 1 - j);
        
        if (furo.type === "ankan") {
          // 暗カン：両端を裏向き、中2枚を表向きで表示
          pais = ["?"].concat(furo.consumed.slice(0, 2)).concat(["?"]);
          poses = [1, 1, 1, 1];  // 全て縦向き
        } else {
          // その他の副露（ポン・チー・大明カン・加カン）
          
          // 相手プレイヤーの方向を計算（1:下家, 2:対面, 3:上家）
          dir = (4 + furo.target - i) % 4;
          
          // 取得牌の位置を決定
          if ((_ref = furo.type) === "daiminkan" || _ref === "kakan") {
            // 大明カン・加カン：4枚のうち取得牌の位置
            laidPos = [null, 3, 1, 0][dir];  // 下家:3, 対面:1, 上家:0
          } else {
            // ポン・チー：3枚のうち取得牌の位置
            laidPos = [null, 2, 1, 0][dir];  // 下家:2, 対面:1, 上家:0
          }
          
          // 牌配列と向きを設定
          pais = furo.consumed.concat([]);  // 消費牌をコピー
          poses = [1, 1, 1];  // 基本は全て縦向き
          
          // 取得牌を指定位置に挿入
          [].splice.apply(pais, [laidPos, laidPos - laidPos].concat(_ref1 = [furo.taken])), _ref1;
          // 取得牌は横向き（pose=3）に設定
          [].splice.apply(poses, [laidPos, laidPos - laidPos].concat(_ref2 = [3])), _ref2;
        }
        
        // 副露を描画
        renderPais(pais, furoView.pais, poses);
        --j;
      }
    }
  }
  
  // 王牌（ワンパイ）の描画
  // 基本は6枚全て裏向き（"?"）で初期化
  wanpais = ["?", "?", "?", "?", "?", "?"];
  
  // ドラ表示牌を配置（3番目の位置から順次表示）
  for (i = _j = 0, _ref3 = action.board.doraMarkers.length; 0 <= _ref3 ? _j < _ref3 : _j > _ref3; i = 0 <= _ref3 ? ++_j : --_j) {
    wanpais[i + 2] = action.board.doraMarkers[i];  // インデックス2から開始
  }
  
  // 王牌を描画
  return renderPais(wanpais, Dytem.wanpais);
};

/**
 * 現在選択されている局オブジェクトを取得する関数
 * @returns {Object} 現在の局オブジェクト
 */
getCurrentKyoku = function() {
  return kyokus[currentKyokuId];
};

/**
 * 現在選択されているアクションを描画する関数
 * getCurrentKyoku()とrenderAction()を組み合わせた便利関数
 * @returns {void} renderAction()の戻り値
 */
renderCurrentAction = function() {
  return renderAction(getCurrentKyoku().actions[currentActionId]);
};

/**
 * 次のアクションに進むナビゲーション関数
 * 最後のアクションに達している場合は何もしない
 * @returns {void}
 */
goNext = function() {
  // 現在の局の最後のアクションかチェック
  if (currentActionId === getCurrentKyoku().actions.length - 1) {
    return;  // 最後の場合は何もしない
  }
  
  ++currentActionId;  // アクションIDを1つ進める
  $("#action-id-label").val(currentActionId);  // UI表示を更新
  return renderCurrentAction();  // 新しいアクションを描画
};

/**
 * 前のアクションに戻るナビゲーション関数
 * 最初のアクションの場合は何もしない
 * @returns {void}
 */
goBack = function() {
  // 最初のアクションかチェック
  if (currentActionId === 0) {
    return;  // 最初の場合は何もしない
  }
  
  --currentActionId;  // アクションIDを1つ戻す
  $("#action-id-label").val(currentActionId);  // UI表示を更新
  return renderCurrentAction();  // 前のアクションを描画
};

/**
 * DOM読み込み完了後の初期化処理
 * イベントハンドラーの登録、UI要素の生成、データの読み込みを実行
 */
$(function() {
  var action, bakazeStr, honba, i, j, kyokuNum, label, playerInfoView, playerView, _i, _j, _k, _l, _len, _ref;
  
  // マウスホイールイベントの設定：前後のアクション移動
  $(window).bind("mousewheel", function(e) {
    e.preventDefault();  // デフォルトのスクロール動作を防止
    if (e.originalEvent.wheelDelta < 0) {
      return goNext();  // 下方向スクロール：次のアクション
    } else if (e.originalEvent.wheelDelta > 0) {
      return goBack();  // 上方向スクロール：前のアクション
    }
  });
  
  // ナビゲーションボタンのイベントハンドラー設定
  $("#prev-button").click(goBack);     // 「Prev」ボタン
  $("#next-button").click(goNext);     // 「Next」ボタン
  
  // 「Go」ボタン：入力されたアクション番号にジャンプ
  $("#go-button").click(function() {
    currentActionId = parseInt($("#action-id-label").val());
    return renderCurrentAction();
  });
  
  // 局選択ドロップダウンの変更イベント
  $("#kyokuSelector").change(function() {
    currentKyokuId = parseInt($("#kyokuSelector").val());
    currentActionId = 0;  // 新しい局では最初のアクションから開始
    return renderCurrentAction();
  });
  
  // 視点変更ボタン：4つの視点を循環
  $("#viewpoint-button").click(function() {
    currentViewpoint = (currentViewpoint + 1) % 4;  // 0→1→2→3→0...
    return renderCurrentAction();
  });
  
  // 全アクションデータの読み込みと処理
  // allActionsはindex.htmlで定義されたグローバル変数
  for (_i = 0, _len = allActions.length; _i < _len; _i++) {
    action = allActions[_i];
    loadAction(action);  // 各アクションを処理して局データを構築
  }
  
  // Dytemテンプレートエンジンの初期化
  Dytem.init();
  
  // 4人のプレイヤー表示要素を動的生成
  for (i = _j = 0; _j < 4; i = ++_j) {
    // プレイヤー表示エリアを追加
    playerView = Dytem.players.append();
    playerView.addClass("player-" + i);  // 位置別CSSクラスを追加
    
    // 各プレイヤーの河（捨て牌）を3行で表示するため、3つの行要素を追加
    for (j = _k = 0; _k < 3; j = ++_k) {
      playerView.hoRows.append();
    }
    
    // プレイヤー情報テーブルの行を追加
    playerInfoView = Dytem.playerInfos.append();
    playerInfoView.index.text(i);                    // プレイヤー番号
    playerInfoView.name.text(playerInfos[i].name);   // プレイヤー名
  }
  
  // 局選択ドロップダウンの選択肢を生成
  for (i = _l = 0, _ref = kyokus.length; 0 <= _ref ? _l < _ref : _l > _ref; i = 0 <= _ref ? ++_l : --_l) {
    bakazeStr = BAKAZE_TO_STR[kyokus[i].bakaze];  // 場風の日本語表記
    honba = kyokus[i].honba;                      // 本場数
    kyokuNum = kyokus[i].kyokuNum;                // 局数
    
    // "東1局 0本場" 形式のラベルを生成
    label = "" + bakazeStr + kyokuNum + "局 " + honba + "本場";
    $("#kyokuSelector").get(0).options[i] = new Option(label, i);
  }
  
  // 初期化完了をコンソールに出力
  console.log("loaded");
  
  // 最初のアクションを描画
  return renderCurrentAction();
});
