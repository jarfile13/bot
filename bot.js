const { Bot, InlineKeyboard } = require("grammy");
const { Connection, Keypair, VersionedTransaction, PublicKey } = require("@solana/web3.js");
const https = require("https");

const BOT_TOKEN = process.env.BOT_TOKEN;
const WALLET_PRIVATE_KEY = JSON.parse(process.env.WALLET_PRIVATE_KEY);
const RPC_ENDPOINT = process.env.RPC_ENDPOINT;
const ALLOWED_CHAT_ID = Number(process.env.ALLOWED_CHAT_ID);
const APIFY_TOKEN = process.env.APIFY_TOKEN;

const bot = new Bot(BOT_TOKEN);
const connection = new Connection(RPC_ENDPOINT, "confirmed");
const wallet = Keypair.fromSecretKey(Uint8Array.from(WALLET_PRIVATE_KEY));

let isRunning = false;
let currentStep = null;
let activePosition = null;
let isPaused = false; 

let config = {
    amountSol: 0.005,
    takeProfitPercent: 10,   
    stopLossPercent: 20,     
    maxSlippage: 10,
    testBuyAmountSol: 0.0005 
};

function request(url, options = {}) {
    return new Promise((resolve, reject) => {
        const req = https.request(url, options, (res) => {
            let data = "";
            res.on("data", (chunk) => data += chunk);
            res.on("end", () => {
                try { resolve(JSON.parse(data)); }
                catch (e) { reject(e); }
            });
        });
        req.on("error", reject);
        if (options.body) req.write(JSON.stringify(options.body));
        req.end();
    });
}

function checkAccess(ctx) {
    return ctx.chat.id === ALLOWED_CHAT_ID;
}

async function isTokenSafe(tokenMint) {
    try {
        const apifyUrl = `https://api.apify.com/v2/acts/ninhothedev~rugcheck-solana-scraper/run-sync-get-dataset-items?token=${APIFY_TOKEN}`;
        const rugRes = await request(apifyUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: { mints: [tokenMint] }
        });

        if (!rugRes || !rugRes.length || !rugRes[0]) {
            await bot.api.sendMessage(ALLOWED_CHAT_ID, "RugCheck didn't respond. Skipping.");
            return false;
        }

        const report = rugRes[0];
        
        if (report.mintAuthority !== null && report.mintAuthority !== "null") {
            await bot.api.sendMessage(ALLOWED_CHAT_ID, `Mint Authority not revoked. Skipping.`);
            return false;
        }
        if (report.freezeAuthority !== null && report.freezeAuthority !== "null") {
            await bot.api.sendMessage(ALLOWED_CHAT_ID, `Freeze Authority not revoked. Skipping.`);
            return false;
        }

        if (report.topHoldersPercent > 30) {
            await bot.api.sendMessage(ALLOWED_CHAT_ID, `Top-10 hold ${report.topHoldersPercent}%. Skipping.`);
            return false;
        }

        if (report.lpLockedPercent < 80 && report.lpBurnedPercent < 80) {
            await bot.api.sendMessage(ALLOWED_CHAT_ID, `Liquidity not locked. Skipping.`);
            return false;
        }

        await bot.api.sendMessage(ALLOWED_CHAT_ID, `RugCheck passed. Testing honeypot...`);
        return true;

    } catch (e) {
        await bot.api.sendMessage(ALLOWED_CHAT_ID, `Check error: ${e.message}. Skipping.`);
        return false;
    }
}

async function testSellability(tokenMint, chatId) {
    try {
        const testLamports = Math.floor(config.testBuyAmountSol * 1000000000);
        
        const buyQuote = await getQuote(
            "So11111111111111111111111111111111111111112",
            tokenMint,
            testLamports
        );
        if (!buyQuote) {
            await bot.api.sendMessage(chatId, "No buy quote. Skipping.");
            return false;
        }

        const buySuccess = await executeSwap(buyQuote, chatId);
        if (!buySuccess) {
            await bot.api.sendMessage(chatId, "Test buy failed.");
            return false;
        }

        const tokenBalance = await getTokenBalance(tokenMint);
        if (!tokenBalance || tokenBalance === "0") {
            await bot.api.sendMessage(chatId, "Failed to get token balance.");
            return false;
        }

        const sellQuote = await getQuote(
            tokenMint,
            "So11111111111111111111111111111111111111112",
            tokenBalance
        );
        if (!sellQuote) {
            await bot.api.sendMessage(chatId, "HONEYPOT! No sell quote. Skipping.");
            return false;
        }

        const sellSuccess = await executeSwap(sellQuote, chatId);
        if (!sellSuccess) {
            await bot.api.sendMessage(chatId, "HONEYPOT! Sell failed. Skipping.");
            return false;
        }

        await bot.api.sendMessage(chatId, `Honeypot test passed. Token can be sold.`);
        return true;

    } catch (e) {
        await bot.api.sendMessage(chatId, `Test error: ${e.message}. Skipping.`);
        return false;
    }
}

async function getQuote(inputMint, outputMint, amount) {
    try {
        const url = `https://api.jup.ag/swap/v1/quote?inputMint=${inputMint}&outputMint=${outputMint}&amount=${amount}&slippageBps=${Math.floor(config.maxSlippage * 100)}`;
        const res = await request(url);
        return res.outAmount ? res : null;
    } catch { return null; }
}

async function executeSwap(quote, chatId) {
    try {
        const swapRes = await request("https://api.jup.ag/swap/v1/swap", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: {
                quoteResponse: quote,
                userPublicKey: wallet.publicKey.toString(),
                wrapAndUnwrapSol: true,
                prioritizationFeeLamports: 5000000
            }
        });
        
        if (!swapRes.swapTransaction) return false;
        
        const txBuffer = Buffer.from(swapRes.swapTransaction, "base64");
        const transaction = VersionedTransaction.deserialize(txBuffer);
        transaction.sign([wallet]);
        
        const txid = await connection.sendTransaction(transaction, { skipPreflight: true, maxRetries: 3 });
        const confirmation = await connection.confirmTransaction(txid, "confirmed");
        
        if (confirmation.value.err) return false;
        return true;
    } catch { return false; }
}

async function getTokenBalance(tokenMint) {
    try {
        const accounts = await connection.getParsedTokenAccountsByOwner(wallet.publicKey, { mint: new PublicKey(tokenMint) });
        if (accounts.value.length === 0) return "0";
        return accounts.value[0].account.data.parsed.info.tokenAmount.amount;
    } catch { return "0"; }
}

async function scanNewLiquidityPool() {
    try {
        const res = await request("https://api.geckoterminal.com/api/v2/networks/solana/new_pools", {
            headers: { "Accept": "application/json;version=20230302", "User-Agent": "Mozilla/5.0" }
        });
        if (!res.data || res.data.length === 0) return null;
        const latestPool = res.data[0];
        const reserveInUsd = parseFloat(latestPool.attributes.reserve_in_usd);
        if (reserveInUsd < 5000 || reserveInUsd > 50000) return null;
        const baseTokenData = latestPool.relationships?.base_token?.data;
        if (!baseTokenData || !baseTokenData.id) return null;
        return baseTokenData.id.split("_")[1];
    } catch { return null; }
}

async function fetchCurrentPrice(tokenMint) {
    try {
        const url = `https://api.jup.ag/swap/v1/quote?inputMint=${tokenMint}&outputMint=So11111111111111111111111111111111111111112&amount=100000000&slippageBps=50`;
        const res = await request(url);
        return res.outAmount ? parseFloat(res.outAmount) / 100000000 : null;
    } catch { return null; }
}

async function runSniperLoop(chatId) {
    while (isRunning && !isPaused) {
        try {
            if (activePosition) {
                await checkPositionStatus(chatId);
            } else {
                const targetToken = await scanNewLiquidityPool();
                if (targetToken) {
                    await bot.api.sendMessage(chatId, `Token found: ${targetToken}`);

                    const safe = await isTokenSafe(targetToken);
                    if (!safe) continue;

                    const sellable = await testSellability(targetToken, chatId);
                    if (!sellable) continue;

                    const lamports = Math.floor(config.amountSol * 1000000000);
                    const quote = await getQuote("So11111111111111111111111111111111111111112", targetToken, lamports);
                    if (!quote) continue;

                    const buySuccess = await executeSwap(quote, chatId);
                    if (buySuccess) {
                        const price = await fetchCurrentPrice(targetToken);
                        activePosition = { mint: targetToken, buyPrice: price, amount: config.amountSol };
                        await bot.api.sendMessage(chatId, `Position opened: ${price}`);
                    }
                }
            }
        } catch (error) {
            await bot.api.sendMessage(chatId, `Error: ${error.message}`);
        }
        await new Promise(res => setTimeout(res, 5000));
    }
}

async function checkPositionStatus(chatId) {
    const currentPrice = await fetchCurrentPrice(activePosition.mint);
    if (!currentPrice) return;
    const change = ((currentPrice - activePosition.buyPrice) / activePosition.buyPrice) * 100;
    
    if (change >= config.takeProfitPercent) {
        await bot.api.sendMessage(chatId, `TP +${change.toFixed(2)}%. Selling...`);
        await sellPosition(activePosition.mint, chatId);
        activePosition = null;
        return;
    }
    
    if (change <= -config.stopLossPercent) {
        await bot.api.sendMessage(chatId, `SL -${change.toFixed(2)}%. Selling...`);
        await sellPosition(activePosition.mint, chatId);
        activePosition = null;
        return;
    }
    
    await bot.api.sendMessage(chatId, `PnL: ${change.toFixed(2)}%`);
}

async function sellPosition(tokenMint, chatId) {
    const balance = await getTokenBalance(tokenMint);
    if (balance === "0") return;
    const quote = await getQuote(tokenMint, "So11111111111111111111111111111111111111112", balance);
    if (!quote) {
        await bot.api.sendMessage(chatId, "No sell quote! Token cannot be sold.");
        return;
    }
    const success = await executeSwap(quote, chatId);
    await bot.api.sendMessage(chatId, success ? "Sold." : "Sell error.");
}

function getMainKeyboard() {
    const statusText = isRunning ? "ACTIVE" : "PAUSED";
    return new InlineKeyboard()
        .text(statusText, "noop").row()
        .text(isRunning ? "STOP BOT" : "START BOT", "toggle_bot").row()
        .text("SOL Amount", "set_amount")
        .text("Take Profit %", "set_tp").row()
        .text("Stop Loss %", "set_sl")
        .text("Slippage %", "set_slippage").row()
        .text("Show Config", "view_config");
}

bot.command("start", async (ctx) => {
    if (!checkAccess(ctx)) return;
    await ctx.reply("Solana Sniper (Safe Mode)", { reply_markup: getMainKeyboard() });
});

bot.callbackQuery("noop", async (ctx) => { await ctx.answerCallbackQuery(); });

bot.callbackQuery("toggle_bot", async (ctx) => {
    if (!checkAccess(ctx)) return;
    isRunning = !isRunning;
    isPaused = !isRunning;
    await ctx.answerCallbackQuery();
    await ctx.editMessageText("Solana Sniper (Safe Mode)", { reply_markup: getMainKeyboard() });
    if (isRunning) {
        await ctx.reply("Scanning... (with checks)");
        runSniperLoop(ctx.chat.id);
    } else {
        await ctx.reply("Stopped. Open position not auto-sold.");
    }
});

bot.callbackQuery("view_config", async (ctx) => {
    if (!checkAccess(ctx)) return;
    await ctx.answerCallbackQuery();
    await ctx.reply(`Settings:\nSOL: ${config.amountSol}\nTP: +${config.takeProfitPercent}%\nSL: -${config.stopLossPercent}%\nSlippage: ${config.maxSlippage}%\n\nTest buy honeypot: ${config.testBuyAmountSol} SOL`);
});

bot.callbackQuery(/set_(amount|tp|sl|slippage)/, async (ctx) => {
    if (!checkAccess(ctx)) return;
    currentStep = ctx.match[1];
    await ctx.answerCallbackQuery();
    await ctx.reply(`Enter value for ${currentStep}:`);
});

bot.on("message:text", async (ctx) => {
    if (!checkAccess(ctx) || !currentStep) return;
    const val = parseFloat(ctx.text);
    if (isNaN(val) || val <= 0) { await ctx.reply("Need a number > 0."); return; }
    if (currentStep === "amount") config.amountSol = val;
    if (currentStep === "tp") config.takeProfitPercent = val;
    if (currentStep === "sl") config.stopLossPercent = val;
    if (currentStep === "slippage") config.maxSlippage = val;
    currentStep = null;
    await ctx.reply("Saved.", { reply_markup: getMainKeyboard() });
});

bot.catch((err) => console.error("Bot error:", err));
bot.start();
console.log("Bot started.");
