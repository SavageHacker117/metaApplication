document.getElementById('mintBtn').onclick = function() {
  let asset = document.getElementById('assetPath').value;
  let meta = document.getElementById('meta').value;
  let os = document.getElementById('os').checked;
  if(window.rubyEngine)
    window.rubyEngine.plugins.nft_creator_market.mint_nft("user123", asset, meta, os);
};

document.getElementById('listBtn').onclick = function() {
  let id = document.getElementById('nftId').value;
  let price = parseFloat(document.getElementById('price').value);
  if(window.rubyEngine)
    window.rubyEngine.plugins.nft_creator_market.list_nft(id, price);
};

document.getElementById('sellBtn').onclick = function() {
  let id = document.getElementById('nftId').value;
  let buyer = document.getElementById('buyer').value;
  if(window.rubyEngine)
    window.rubyEngine.plugins.nft_creator_market.sell_nft(id, buyer);
};

document.getElementById('statusBtn').onclick = function() {
  if(window.rubyEngine) {
    let list = window.rubyEngine.plugins.nft_creator_market.marketplace_status();
    document.getElementById('nftLog').innerHTML =
      "<pre>" + JSON.stringify(list, null, 2) + "</pre>";
  }
};
