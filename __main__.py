from __params__ import MODEL, CRAWL_PLATFORM, SKIP_TRAINING, SKIP_PREDICTION

if CRAWL_PLATFORM:
    from asyncio import run
    from src.crawl import crawl_twitter, crawl_youtube, crawl_bluesky

    if CRAWL_PLATFORM == "twitter":
        run(crawl_twitter())
    elif CRAWL_PLATFORM == "youtube":
        crawl_youtube()
    elif CRAWL_PLATFORM == "bluesky":
        crawl_bluesky()
else:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from src import preprocess, train, evaluate, predict, visualize

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL,
                                                               num_labels=3)

    training, validation, testing = preprocess(tokenizer)

    if not SKIP_TRAINING:
        trainer = train(model, tokenizer, training, validation)
        results = evaluate(trainer, testing)
        print(results)

    for platform in ["twitter", "youtube", "bluesky"]:
        if not SKIP_PREDICTION:
            predict(platform)
        visualize(f"{platform}-predictions")
