const Parse = require('parse/node');

Parse.initialize(
  process.env.B4A_APP_ID,
  process.env.B4A_JS_KEY,
  process.env.B4A_MASTER_KEY
);
Parse.serverURL = 'https://parseapi.back4app.com';

async function getAll() {
  const query = new Parse.Query('Knowledge');
  query.limit(10000);
  const results = await query.find({ useMasterKey: true });
  const map = {};
  for (const obj of results) {
    map[obj.get('question')] = obj.get('answer');
  }
  return map;
}

async function upsert(question, answer) {
  const query = new Parse.Query('Knowledge');
  query.equalTo('question', question);
  let obj = await query.first({ useMasterKey: true });
  if (!obj) {
    obj = new Parse.Object('Knowledge');
    obj.set('question', question);
  }
  obj.set('answer', answer);
  await obj.save(null, { useMasterKey: true });
}

async function remove(question) {
  const query = new Parse.Query('Knowledge');
  query.equalTo('question', question);
  const obj = await query.first({ useMasterKey: true });
  if (obj) await obj.destroy({ useMasterKey: true });
}

module.exports = { getAll, upsert, remove };
